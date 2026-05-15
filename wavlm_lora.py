"""
WavLM Large (full finetuning) → Speech Enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel

from datasets import ID2CHAR, BLANK_ID


# ─────────────────────────────────────────────────────────────────────────────
# Attentive Statistical Pooling
# ─────────────────────────────────────────────────────────────────────────────

class AttentiveStatisticalPooling(nn.Module):
    """
    Input  : (B, T, D)
    Output : (B, 2*D)  — concat of weighted mean + weighted std
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        e = self.attention(x)                          # (B, T, 1)
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            e = e.masked_fill(mask.unsqueeze(-1), float("-inf"))
        alpha = F.softmax(e, dim=1)                    # (B, T, 1)
        mu    = (alpha * x).sum(dim=1)                 # (B, D)
        var   = (alpha * x.pow(2)).sum(dim=1) - mu.pow(2)
        sigma = var.clamp(min=1e-8).sqrt()             # (B, D)
        return torch.cat([mu, sigma], dim=-1)          # (B, 2*D)


# ─────────────────────────────────────────────────────────────────────────────
# Main SE model
# ─────────────────────────────────────────────────────────────────────────────

class WavLMLoRASE(nn.Module):
    """WavLM Large (all weights trainable) for speech enhancement."""

    WAVLM_HOP = 320   # samples per output frame

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg            = cfg
        self.content_layer  = cfg["wavlm"]["content_layer"]
        self.acoustic_layer = cfg["wavlm"]["acoustic_layer"]

        # ── WavLM: load pretrained; CNN feature extractor frozen ─────────────
        self.wavlm = WavLMModel.from_pretrained(cfg["wavlm"]["model_name"])
        # if cfg["wavlm"].get("freeze_cnn", True):
        #     self.wavlm.freeze_feature_encoder()

        hidden_dim = cfg["wavlm"]["output_dim"]   # 1024

        # ── Speaker head (on acoustic_layer output) ───────────────────────────
        embed_dim    = cfg.get("sv", {}).get("embed_dim", 192)
        pool_hidden  = cfg.get("pooling", {}).get("hidden_dim", 128)
        self.asp      = AttentiveStatisticalPooling(hidden_dim, hidden_dim=pool_hidden)
        self.spk_proj = nn.Linear(hidden_dim * 2, embed_dim)
        self.spk_bn   = nn.BatchNorm1d(embed_dim)

        # ── CTC head (on content_layer output, full resolution) ──────────────
        vocab_size       = cfg["ctc"]["vocab_size"]
        self.ctc_proj    = nn.Linear(hidden_dim, vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(
            blank=cfg["ctc"]["blank_id"], zero_infinity=True
        )

        self.proj_content  = nn.Linear(vocab_size, hidden_dim)
        self.proj_acoustic = nn.Linear(hidden_dim, hidden_dim)

        # ── Mel head (direct mel prediction, used by no-transformer variant) ──
        n_mels = cfg.get("n_mels", 80)
        self.mel_head = nn.Linear(hidden_dim, n_mels)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_content(self, hidden_states: tuple) -> torch.Tensor:
        """content_layer가 int이면 단일 레이어, list이면 해당 구간 sum."""
        if isinstance(self.content_layer, (list, tuple)):
            stacked = torch.stack(
                hidden_states[self.content_layer[0] : self.content_layer[-1] + 1], dim=0
            )  # (num_layers, B, T, 1024)
            return stacked.sum(dim=0)   # (B, T, 1024)
        return hidden_states[self.content_layer]  # (B, T, 1024)

    def frame_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        """Exact WavLM CNN output frame count per utterance."""
        return self.wavlm._get_feat_extract_output_lengths(audio_lengths)

    # ── Inference encode (no loss computation) ───────────────────────────────

    @torch.no_grad()
    def encode(
        self,
        wav:           torch.Tensor,   # (B, T_audio)
        audio_lengths: torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T         = wav.shape[1]
        attn_mask = (torch.arange(T, device=wav.device).unsqueeze(0)
                     < audio_lengths.unsqueeze(1)).long()
        out           = self.wavlm(wav, attention_mask=attn_mask,
                                   output_hidden_states=True)
        hidden_states = out.hidden_states

        content_emb = self._extract_content(hidden_states)  # (B, T, 1024)
        T_frames    = content_emb.size(1)
        frame_lens  = self.frame_lengths(audio_lengths).clamp(max=T_frames)

        return content_emb, frame_lens

    # ── Forward (training) ───────────────────────────────────────────────────

    def forward(
        self,
        noisy_wav:     torch.Tensor,            # (B, T_audio)
        audio_lengths: torch.Tensor,            # (B,)
        text_ids:      torch.Tensor | None = None,
        text_lengths:  torch.Tensor | None = None,
    ):
        T         = noisy_wav.shape[1]
        attn_mask = (torch.arange(T, device=noisy_wav.device).unsqueeze(0)
                     < audio_lengths.unsqueeze(1)).long()
        out           = self.wavlm(noisy_wav, attention_mask=attn_mask, output_hidden_states=True)
        hidden_states = out.hidden_states               # tuple of 25

        content_emb = self._extract_content(hidden_states)  # (B, T, 1024)
        T_frames    = content_emb.size(1)
        frame_lens  = self.frame_lengths(audio_lengths).clamp(min=1, max=T_frames)

        if text_ids is not None and text_lengths is not None:
            logits   = self.ctc_proj(content_emb)
            log_prob = F.log_softmax(logits, dim=-1).permute(1, 0, 2).clamp(min=-100.0)
            ctc_loss = self.ctc_loss_fn(log_prob, text_ids, frame_lens, text_lengths)
        else:
            ctc_loss = torch.zeros((), device=noisy_wav.device)

        return hidden_states[-1], ctc_loss, frame_lens

    # ── Greedy CTC decode ─────────────────────────────────────────────────────

    @torch.no_grad()
    def decode_greedy(
        self,
        content_emb: torch.Tensor,   # (B, T, 1024)
        frame_lens:  torch.Tensor,   # (B,)
    ) -> list[str]:
        ids_batch = self.ctc_proj(content_emb).argmax(-1)   # (B, T)
        blank     = self.ctc_loss_fn.blank
        results   = []
        for b, length in enumerate(frame_lens):
            seq       = ids_batch[b, :length].tolist()
            collapsed = []
            prev      = -1
            for s in seq:
                if s != prev:
                    if s != blank:
                        collapsed.append(s)
                    prev = s
            results.append("".join(ID2CHAR.get(i, "?") for i in collapsed))
        return results

    # ── Parameter summary ─────────────────────────────────────────────────────

    def param_summary(self) -> dict:
        trainable = frozen = wavlm_p = head_p = 0
        for name, p in self.named_parameters():
            n = p.numel()
            if p.requires_grad:
                trainable += n
                if name.startswith("wavlm."):
                    wavlm_p += n
                else:
                    head_p += n
            else:
                frozen += n
        for buf in self.buffers():
            if buf is not None:
                frozen += buf.numel()
        return {"trainable": trainable, "frozen": frozen,
                "wavlm": wavlm_p, "heads": head_p}
