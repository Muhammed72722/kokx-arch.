"""
Aşama 1 — Mimari Probe Konfigürasyonları

Üç mimariyi AYNI iskelet (RMSNorm + SwiGLU MLP), AYNI boyut (dim/n_layers),
AYNI veri ve AYNI eğitim tarifiyle yarıştırıyoruz. Tek değişken: sequence-mixer.

  CONFIG_A — Mamba-2-Hybrid, düşük attention (~7%)
             Kaynak: Nvidia'nın "An Empirical Study of Mamba-based LMs" tarifi
             (24 mamba2 : 4 attn : 28 mlp oranının küçültülmüş hali)

  CONFIG_B — Mamba-2-Hybrid, yüksek attention (~25%)
             Kaynak: Jamba'nın 1:3 (attn:mamba) ucu — kod'un ihtiyaç duyduğu
             in-context copying/recall için daha fazla attention

  CONFIG_C — Gated DeltaNet Hybrid (~25% attention)
             Kaynak: Qwen3.5 mimarisi (24 GDN : 8 attn ~ 3:1) — senin zaten
             fine-tune ettiğin ailenin backbone'u, kod odaklı ayarlarda
             bazı çalışmalarda Mamba-2'yi geçiyor

Hepsi ~70-90M parametre (non-embedding) civarında — Kaggle T4/P100'de
birkaç günde 3-5B token'lık probe koşusu için pratik boyut.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 49152          # StarCoder2 tokenizer ile hizalı, train_probe.py'de override edilir
    dim: int = 640
    n_layers: int = 24
    layer_pattern: list = field(default_factory=list)
    mixer_kwargs: dict = field(default_factory=dict)
    max_seq_len: int = 2048


def build_layer_pattern(n_layers: int, attn_ratio: float, other: str = "mamba2") -> list:
    """attn katmanlarını eşit aralıklarla dağıtır, geri kalanı `other` ile doldurur."""
    n_attn = max(1, round(n_layers * attn_ratio))
    pattern = [other] * n_layers
    step = n_layers / n_attn
    for i in range(n_attn):
        idx = min(n_layers - 1, round(i * step))
        pattern[idx] = "attn"
    return pattern


DIM = 640
N_LAYERS = 24

CONFIG_A_MAMBA_LOW_ATTN = ModelConfig(
    dim=DIM,
    n_layers=N_LAYERS,
    layer_pattern=build_layer_pattern(N_LAYERS, attn_ratio=0.08, other="mamba2"),
    mixer_kwargs={
        "mamba2": {"d_state": 128, "d_conv": 4, "expand": 2},
        "attn": {"n_heads": 10, "n_kv_heads": 2, "max_seq_len": 2048},
    },
)

CONFIG_B_MAMBA_HIGH_ATTN = ModelConfig(
    dim=DIM,
    n_layers=N_LAYERS,
    layer_pattern=build_layer_pattern(N_LAYERS, attn_ratio=0.25, other="mamba2"),
    mixer_kwargs={
        "mamba2": {"d_state": 128, "d_conv": 4, "expand": 2},
        "attn": {"n_heads": 10, "n_kv_heads": 2, "max_seq_len": 2048},
    },
)

CONFIG_C_GDN_HYBRID = ModelConfig(
    dim=DIM,
    n_layers=N_LAYERS,
    layer_pattern=build_layer_pattern(N_LAYERS, attn_ratio=0.25, other="gdn"),
    mixer_kwargs={
        # head_dim'i EXPLICIT vermek şart: fla varsayılanı head_dim=256, hidden_size'dan
        # bağımsız sabit bir değer. dim=640, num_heads=10 iken head_dim=64 vermezsek
        # key/value projeksiyonları 640 yerine 2560/5120'ye şişiyor (~4x parametre şişmesi).
        "gdn": {"num_heads": 10, "head_dim": 64},
        "attn": {"n_heads": 10, "n_kv_heads": 2, "max_seq_len": 2048},
    },
)

PROBE_CONFIGS = {
    "A_mamba_low_attn": CONFIG_A_MAMBA_LOW_ATTN,
    "B_mamba_high_attn": CONFIG_B_MAMBA_HIGH_ATTN,
    "C_gdn_hybrid": CONFIG_C_GDN_HYBRID,
}

if __name__ == "__main__":
    for name, cfg in PROBE_CONFIGS.items():
        print(f"{name}: {cfg.layer_pattern}")
        counts = {t: cfg.layer_pattern.count(t) for t in set(cfg.layer_pattern)}
        print(f"  -> {counts}")
