"""
conformer_decoder.py — Conformer-based latent feature decoder.

Architecture
────────────
  Given:
    content_emb  (B, T, content_dim=1024)  — noisy WavLM + MoE last-layer output
    spk_emb      (B, spk_dim=192)          — utterance-level speaker embedding

  Step 1 · Query projection  (ctc_dim + spk_dim → content_dim)
    spk_exp  = spk_emb.unsqueeze(1).expand(B, T, spk_dim)
    q_feat   = LeakyReLU(Linear(concat(content_fc, spk_exp)))   → (B, T, content_dim)

  Step 2 · Cross-attention
    Q = q_feat                     (B, T, content_dim)  — projected CTC logits + spk_emb
    K = V = content_emb            (B, T, content_dim)  — WavLM+MoE last-layer output
    attn_out = weighted sum of V   (B, T, content_dim)  — output dim = V dim ✓
    x = LayerNorm(attn_out)                             → (B, T, content_dim)  [no projection]

  Step 3 · Sinusoidal positional encoding
    x = x + SinusoidalPE(T, d_model)

  Step 4 · N × ConformerBlock
    Each block (residuals explicit at block level):
      x = x + 0.5 * FFN₁(x)
      x = x + MHSA(x)
      x = x + Conv(x)
      x = x + 0.5 * FFN₂(x)
      x = LayerNorm(x)

  Step 5 · Output projection
    out = Linear(d_model → wavlm_dim=1024)                       → (B, T, 1024)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ─────────────────────────────────────────────────────────────────────────────
# Conformer sub-modules  (pure computation, NO residual inside)
# Residual connections are applied explicitly in ConformerBlock.forward().
# ─────────────────────────────────────────────────────────────────────────────

class FeedForwardModule(nn.Module):
    """
    LN → Linear(d→ffn) → Swish → Dropout → Linear(ffn→d) → Dropout
    No residual here — added by ConformerBlock as: x = x + 0.5 * ffn(x)
    """
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1  = nn.Linear(d_model, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(F.silu(self.fc1(self.norm(x))))
        return self.drop(self.fc2(x))


class MultiHeadSelfAttentionModule(nn.Module):
    """
    LN → MHSA → Dropout
    No residual here — added by ConformerBlock as: x = x + mhsa(x)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        return self.drop(h)


class ConvolutionModule(nn.Module):
    """
    LN → PW1(×2) → GLU → DepthwiseConv → BN → Swish → PW2 → Dropout
    No residual here — added by ConformerBlock as: x = x + conv(x)

    PW1: pointwise expand ×2 for GLU gate
    DW : depthwise conv (kernel_size must be odd)
    PW2: pointwise project back
    """
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(d_model)
        self.pw1  = nn.Linear(d_model, 2 * d_model)
        self.dw   = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=padding, groups=d_model, bias=False,
        )
        self.bn   = nn.BatchNorm1d(d_model)
        self.pw2  = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw1(self.norm(x))            # (B, T, 2*d)
        x = F.glu(x, dim=-1)                  # (B, T, d)
        x = x.transpose(1, 2)                 # (B, d, T)
        x = F.silu(self.bn(self.dw(x)))       # (B, d, T)
        x = x.transpose(1, 2)                 # (B, T, d)
        return self.drop(self.pw2(x))


# ─────────────────────────────────────────────────────────────────────────────
# Conformer block  (residuals are explicit here)
# ─────────────────────────────────────────────────────────────────────────────

class ConformerBlock(nn.Module):
    """
    Full Conformer block (Gulati et al., 2020).
    Residuals are applied at this level, not inside the sub-modules:

      x = x + 0.5 * FFN₁(x)
      x = x + MHSA(x)
      x = x + Conv(x)
      x = x + 0.5 * FFN₂(x)
      x = LayerNorm(x)
    """
    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        ffn_dim:     int,
        conv_kernel: int   = 31,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, n_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout)
        self.ffn2 = FeedForwardModule(d_model, ffn_dim, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = x + self.mhsa(x, key_padding_mask=key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# Latent decoder
# ─────────────────────────────────────────────────────────────────────────────

class TransformerDecoder(nn.Module):
    """
    Transformer decoder for the e2e vocoder pipeline.

    Takes concat(content_fc, spk_emb_expanded) as input and outputs vocos-ready
    latent features (d_model dim).  A separate mel_head projects to n_mels for
    L1 supervision during training.

    Parameters
    ----------
    input_dim : content_fc_dim + spk_dim  (default 29 + 192 = 221)
    d_model   : internal + output dimension (must match vocos input_channels, default 1024)
    n_mels    : mel bins for L1 supervision head (default 80)
    """

    def __init__(
        self,
        input_dim: int   = 221,
        d_model:   int   = 1024,
        n_layers:  int   = 4,
        n_heads:   int   = 8,
        ffn_dim:   int   = 2048,
        dropout:   float = 0.1,
        max_len:   int   = 4096,
        n_mels:    int   = 80,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = SinusoidalPE(d_model, max_len=max_len, dropout=dropout)
        self.layers     = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])
        # Mel head: used only for L1 supervision, NOT fed into vocos
        self.mel_head = nn.Linear(d_model, n_mels)

    def forward(
        self,
        x:          torch.Tensor,               # (B, T, input_dim)
        frame_lens: torch.Tensor | None = None, # (B,)
    ) -> torch.Tensor:
        """
        Returns
        -------
        (B, T, d_model)  — vocos-ready latent features
        """
        B, T, _ = x.shape

        key_padding_mask: torch.Tensor | None = None
        if frame_lens is not None:
            idx = torch.arange(T, device=x.device).unsqueeze(0)
            key_padding_mask = idx >= frame_lens.unsqueeze(1)   # (B, T)  True = pad

        x = self.input_proj(x)          # (B, T, d_model)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)

        return x                         # (B, T, d_model)


class LatentDecoder(nn.Module):
    """
    Predicts clean WavLM last-layer latent features from noisy
    WavLM+MoE content embedding and speaker embedding.

    Cross-attention:
      Q = LeakyReLU(Linear(concat(content_fc, spk_emb_expanded)))   (B, T, d_model)
          content_fc  = ctc_proj(content_emb)  — CTC logit values   (B, T, ctc_dim=29)
          spk_emb_exp = spk_emb expanded to (B, T, spk_dim)
      K = V = content_emb                                            (B, T, content_dim=1024)
          ← WavLM+MoE last-layer output directly
          (PyTorch handles internal projection via kdim/vdim)

    Parameters
    ----------
    content_dim  : WavLM last-layer dimension for K/V (default 1024)
    d_model      : internal model dimension
    n_layers     : number of ConformerBlocks
    n_heads      : attention heads (must divide d_model)
    ffn_dim      : FFN inner dimension
    conv_kernel  : depthwise conv kernel size (odd)
    dropout      : dropout probability
    wavlm_dim    : output dimension = clean latent dimension (1024)
    max_len      : maximum sequence length for sinusoidal PE table
    """

    def __init__(
        self,
        content_dim: int   = 1024,
        d_model:     int   = 1024,
        n_layers:    int   = 4,
        n_heads:     int   = 8,
        ffn_dim:     int   = 2048,
        conv_kernel: int   = 31,
        dropout:     float = 0.1,
        wavlm_dim:   int   = 1024,
        max_len:     int   = 4096,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim   = content_dim,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.self_norm = nn.LayerNorm(content_dim)

        # Step 3: sinusoidal positional encoding
        self.pos_enc = SinusoidalPE(d_model, max_len=max_len, dropout=dropout)

        # Step 4: Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, ffn_dim, conv_kernel, dropout)
            for _ in range(n_layers)
        ])

        # Step 5: output projection → clean latent dimension
        self.out_proj = nn.Linear(d_model, wavlm_dim)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        content_emb: torch.Tensor,              # (B, T, content_dim)
        frame_lens:  torch.Tensor | None = None,# (B,)
    ) -> torch.Tensor:
        """
        Returns
        -------
        (B, T, wavlm_dim)  predicted clean WavLM latent features
        """
        B, T, _ = content_emb.shape

        # Padding mask: True = pad position (ignored in attention)
        key_padding_mask: torch.Tensor | None = None
        if frame_lens is not None:
            idx = torch.arange(T, device=content_emb.device).unsqueeze(0)
            key_padding_mask = idx >= frame_lens.unsqueeze(1)   # (B, T)

        attn_out, _ = self.self_attn(
            content_emb, content_emb, content_emb,
            key_padding_mask=key_padding_mask,
        )
        x = self.self_norm(attn_out)                               # (B, T, content_dim)

        # Step 3: Positional encoding
        x = self.pos_enc(x)

        # Step 4: Conformer blocks
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        # Step 5: Output projection
        return self.out_proj(x)                                      # (B, T, wavlm_dim)
