"""
KOK-X Hibrit Mimari — Kod-odaklı SLM için modüler sequence-mixer bloğu.

Desteklenen mixer tipleri:
  - "mamba2"  -> mamba_ssm.Mamba2               (pip install mamba-ssm causal-conv1d)
  - "gdn"     -> fla.layers.GatedDeltaNet        (pip install flash-linear-attention)
  - "attn"    -> Bu dosyadaki CausalSelfAttention (RoPE + GQA, saf PyTorch, bağımlılık yok)

Tasarım Jamba/Nemotron-H tarzını izler: her blok = norm -> mixer -> residual -> norm -> SwiGLU MLP -> residual.
Bu, üç mimariyi de AYNI iskelete oturtup adil karşılaştırma yapmamızı sağlıyor —
tek değişken sequence-mixer, geri kalan her şey (norm, MLP, init, optimizer) sabit.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Ortak bileşenler
# --------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 4.0):
        super().__init__()
        hidden = int(dim * hidden_mult * 2 / 3)
        hidden = 64 * ((hidden + 63) // 64)  # 64'e yuvarla (tensor-core dostu)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def _rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x, cos, sin):
    # x: (B, H, T, Dh)
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack([out1, out2], dim=-1).flatten(-2)
    return out


class CausalSelfAttention(nn.Module):
    """RoPE + Grouped Query Attention. Bağımlılık yok, kod-odaklı in-context/copying
    kabiliyeti için hibritteki 'hafıza' katmanı budur."""

    def __init__(self, dim: int, n_heads: int = 8, n_kv_heads: int = 2, max_seq_len: int = 4096):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = _rope_cache(T, self.head_dim, x.device)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # GQA: kv head'leri q head sayısına genişlet
        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class Mamba2Layer(nn.Module):
    def __init__(self, dim: int, d_state: int = 128, d_conv: int = 4, expand: int = 2, **_):
        super().__init__()
        from mamba_ssm import Mamba2  # pip install mamba-ssm causal-conv1d
        self.mixer = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return self.mixer(x)


class GatedDeltaNetLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, head_dim: int = None, **_):
        super().__init__()
        from fla.layers import GatedDeltaNet  # pip install flash-linear-attention
        # head_dim'i dim/num_heads'e göre EXPLICIT vermek şart -- fla'nın varsayılanı
        # (256) hidden_size'dan bağımsız sabit, verilmezse iç boyut hedeflenenin
        # kat kat üzerine şişer (bkz. configs.py'deki not).
        if head_dim is None:
            head_dim = dim // num_heads
        self.mixer = GatedDeltaNet(hidden_size=dim, num_heads=num_heads, head_dim=head_dim)

    def forward(self, x):
        out = self.mixer(x)
        return out[0] if isinstance(out, tuple) else out


MIXER_REGISTRY = {
    "attn": CausalSelfAttention,
    "mamba2": Mamba2Layer,
    "gdn": GatedDeltaNetLayer,
}


# --------------------------------------------------------------------------
# Hibrit blok ve model
# --------------------------------------------------------------------------

class HybridBlock(nn.Module):
    def __init__(self, dim: int, mixer_type: str, mixer_kwargs: dict):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.mixer = MIXER_REGISTRY[mixer_type](dim, **mixer_kwargs)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class KOKXHybrid(nn.Module):
    """config.layer_pattern: ör. ["mamba2","mamba2","attn","mamba2",...]
    Mamba-2/GDN katmanları pozisyon bilgisini implicit öğrenir (Jamba bulgusu:
    ilk katman mamba/gdn ise ayrı positional embedding gerekmez); attn katmanları
    kendi RoPE'unu taşır."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([
            HybridBlock(config.dim, layer_type, config.mixer_kwargs.get(layer_type, {}))
            for layer_type in config.layer_pattern
        ])
        self.norm_f = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.2, top_p=0.95):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)["logits"][:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_tok = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n
