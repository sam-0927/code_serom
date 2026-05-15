"""
MoE-LoRA modules for WavLM fine-tuning.

MoELoRALinear  — drop-in nn.Linear replacement with Mixture-of-Experts LoRA
inject_moe_lora — injects MoELoRALinear into a WavLMModel in-place

Orthogonality loss  (MoLEx, inter-expert)
──────────────────────────────────────────
  Encourages DIFFERENT experts to span distinct subspaces.

      L_orth = ‖ G − I ‖_F²      G[i,j] = cos(flatten(A_i), flatten(A_j))

Singular-value / rank-utilisation loss  (intra-expert)
────────────────────────────────────────────────────────
  Encourages the r row-vectors of each expert's A to be mutually orthogonal.

      For each expert i:
        Ā_i  = row-wise L2-normalised A_i        (r, d_in)
        G_i  = Ā_i  Ā_i^T                        (r, r)
        L_sv_i = ‖ G_i − I ‖_F²
      L_sv = mean over all experts and all modules

Reference: MoLEx: Mixture of LoRA Experts in Speech Self-Supervised Models
           arXiv:2509.09175
"""

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel


# ─────────────────────────────────────────────────────────────────────────────
# MoE-LoRA linear layer
# ─────────────────────────────────────────────────────────────────────────────

class MoELoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with Mixture-of-Experts LoRA.

    The frozen pretrained weight is stored as a buffer (not a Parameter) so
    it is saved with the model but never updated by the optimizer.

    Forward:
        out = W_pretrained · x  +  (α/r) · Σ_i  w_i(x) · B_i · A_i · x
        w_i(x) = softmax( Router(x) )_i       per-token soft gate
    """

    def __init__(
        self,
        pretrained_weight: torch.Tensor,  # (d_out, d_in) — copied at construction
        pretrained_bias:   torch.Tensor | None = None,
        n_experts: int   = 4,
        rank:      int   = 8,
        alpha:     float = 16.0,
        dropout:   float = 0.05,
    ):
        super().__init__()

        d_out, d_in = pretrained_weight.shape
        self.in_features  = d_in
        self.out_features = d_out
        self.n_experts    = n_experts
        self.rank         = rank
        self.scaling      = alpha / rank

        self.register_buffer("weight", pretrained_weight.detach().clone())
        self.register_buffer(
            "bias",
            pretrained_bias.detach().clone() if pretrained_bias is not None else None,
        )

        # A: down-projection  (E, r, d_in)
        # B: up-projection    (E, d_out, r)  — zero-init → ΔW = 0 at start
        self.lora_A = nn.Parameter(torch.empty(n_experts, rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(n_experts, d_out, rank))

        self.router = nn.Linear(d_in, n_experts, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        for i in range(n_experts):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (..., d_in)  →  (..., d_out)"""
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)           # (N, d_in)

        out  = F.linear(x_2d, self.weight, self.bias)    # (N, d_out)

        gate = F.softmax(self.router(x_2d), dim=-1)      # (N, E)
        # self.last_gate = gate.detach()

        x_drop = self.lora_dropout(x_2d)                 # (N, d_in)
        down   = torch.einsum("ni,eir->ner", x_drop, self.lora_A.permute(0, 2, 1))
        up     = torch.einsum("ner,eor->neo", down, self.lora_B)
        lora_out = (gate.unsqueeze(-1) * up).sum(dim=1)  # (N, d_out)

        out = out + self.scaling * lora_out
        return out.reshape(*orig_shape[:-1], self.out_features)

    def get_ortho_loss(self) -> torch.Tensor:
        """Inter-expert orthogonality loss (MoLEx). L_orth = ‖ G − I ‖_F²"""
        A      = self.lora_A.reshape(self.n_experts, -1)
        A_norm = A / (A.norm(dim=1, keepdim=True) + 1e-8)
        G      = A_norm @ A_norm.T
        I      = torch.eye(self.n_experts, device=A.device, dtype=A.dtype)
        return ((G - I) ** 2).sum()

    def get_sv_loss(self) -> torch.Tensor:
        """Intra-expert rank-utilisation loss. L_sv = mean_i ‖ G_i − I ‖_F²"""
        A      = self.lora_A                                         # (E, r, d_in)
        A_norm = A / (A.norm(dim=2, keepdim=True) + 1e-8)
        G      = torch.bmm(A_norm, A_norm.transpose(1, 2))          # (E, r, r)
        I      = torch.eye(self.rank, device=A.device, dtype=A.dtype).unsqueeze(0)
        return ((G - I) ** 2).sum() / self.n_experts


# ─────────────────────────────────────────────────────────────────────────────
# Injection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _replace_linear(parent: nn.Module, attr: str, cfg: dict) -> bool:
    orig: nn.Linear = getattr(parent, attr, None)
    if orig is None or not isinstance(orig, nn.Linear):
        return False
    moe = MoELoRALinear(
        pretrained_weight = orig.weight.data,
        pretrained_bias   = orig.bias.data if orig.bias is not None else None,
        n_experts = cfg["n_experts"],
        rank      = cfg["rank"],
        alpha     = float(cfg["alpha"]),
        dropout   = float(cfg["dropout"]),
    )
    setattr(parent, attr, moe)
    return True


def _patch_attention_module_calls(wavlm: WavLMModel) -> int:
    """
    WavLM attention bypasses MoELoRALinear.forward() by passing raw weight
    tensors to F.multi_head_attention_forward. This patch replaces each
    layer's torch_multi_head_self_attention with an eager implementation that
    calls self.q_proj(x) etc. as module calls so LoRA deltas are applied.
    Must be called AFTER inject_moe_lora().
    """
    def _eager_mha(
        self_attn,
        hidden_states: torch.Tensor,
        attention_mask,
        gated_position_bias: torch.Tensor,
        output_attentions: bool,
    ):
        bsz, tgt_len, _ = hidden_states.shape
        num_heads = self_attn.num_heads
        head_dim  = self_attn.head_dim
        embed_dim = self_attn.embed_dim

        q = self_attn.q_proj(hidden_states)
        k = self_attn.k_proj(hidden_states)
        v = self_attn.v_proj(hidden_states)

        def split_heads(x):
            return (x.view(bsz, tgt_len, num_heads, head_dim)
                     .transpose(1, 2)
                     .reshape(bsz * num_heads, tgt_len, head_dim))
        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self_attn.scaling
        attn_weights = attn_weights + gated_position_bias

        if attention_mask is not None:
            key_pad = attention_mask.ne(1)
            attn_weights = attn_weights.view(bsz, num_heads, tgt_len, tgt_len)
            attn_weights = attn_weights.masked_fill(key_pad[:, None, None, :], float('-inf'))
            attn_weights = attn_weights.view(bsz * num_heads, tgt_len, tgt_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_out = (attn_weights
                                .view(bsz, num_heads, tgt_len, tgt_len)
                                .mean(1))
        else:
            attn_weights_out = None

        attn_weights = F.dropout(attn_weights, p=self_attn.dropout, training=self_attn.training)
        attn_output  = torch.bmm(attn_weights, v)
        attn_output  = (attn_output
                        .view(bsz, num_heads, tgt_len, head_dim)
                        .transpose(1, 2)
                        .reshape(bsz, tgt_len, embed_dim))
        attn_output  = self_attn.out_proj(attn_output)

        return attn_output, attn_weights_out

    patched = 0
    for layer in wavlm.encoder.layers:
        attn = layer.attention
        attn.torch_multi_head_self_attention = types.MethodType(_eager_mha, attn)
        patched += 1

    print(f"[LoRA] Patched {patched} WavLM attention layers → module-call QKV+out_proj")
    return patched


_ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "out_proj")
_FF_MODULES   = ("intermediate_dense", "output_dense")


def _parse_target_layers(spec, n_layers: int) -> list[int]:
    if spec is None or spec == [] or spec == "":
        return []
    if spec == "all":
        return list(range(n_layers))
    if isinstance(spec, str) and "-" in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    if isinstance(spec, (list, range)):
        return sorted(int(i) for i in spec)
    return []


def inject_moe_lora(wavlm: WavLMModel, lora_cfg: dict) -> int:
    """
    Inject MoE-LoRA into WavLM transformer layers.

        lora_cfg['modules'] = {
            'v_proj':             {'target_layers': '0-7'},
            'intermediate_dense': {'target_layers': '16-23'},
            ...
        }

    Returns the total number of projections replaced.
    """
    layers   = wavlm.encoder.layers
    n_layers = len(layers)

    module_layers: dict[str, set[int]] = {}
    for mname, mcfg in lora_cfg["modules"].items():
        parsed = _parse_target_layers(mcfg.get("target_layers", []), n_layers)
        module_layers[mname] = set(parsed)

    injected = 0
    needs_attn_patch = False

    for li, layer in enumerate(layers):
        for mname in _ATTN_MODULES:
            if li in module_layers.get(mname, set()):
                if _replace_linear(layer.attention, mname, lora_cfg):
                    injected += 1
                    needs_attn_patch = True
        for mname in _FF_MODULES:
            if li in module_layers.get(mname, set()):
                if _replace_linear(layer.feed_forward, mname, lora_cfg):
                    injected += 1

    print(f"[LoRA] Injected MoE-LoRA into {injected} projections  "
          f"n_experts={lora_cfg['n_experts']}  rank={lora_cfg['rank']}")
    for mname in list(_ATTN_MODULES) + list(_FF_MODULES):
        ls = sorted(module_layers.get(mname, set()))
        if ls:
            print(f"  {mname:<22}: layers {ls[0]}–{ls[-1]}  ({len(ls)} layers)")

    if needs_attn_patch:
        _patch_attention_module_calls(wavlm)

    return injected
