"""Her config için gerçek parametre sayısını doğrula.
Not: mamba-ssm / flash-linear-attention kurulu olmadan bu script çalışmaz
çünkü mixer'lar gerçek kütüphaneleri import ediyor (fake sayı üretmiyoruz)."""

from hybrid_model import KOKXHybrid
from configs import PROBE_CONFIGS

if __name__ == "__main__":
    for name, cfg in PROBE_CONFIGS.items():
        cfg.vocab_size = 49152  # StarCoder2 tokenizer varsayımı, gerçek koşuda override edilir
        model = KOKXHybrid(cfg)
        total = sum(p.numel() for p in model.parameters())
        non_emb = model.num_params(non_embedding=True)
        print(f"{name}: toplam {total/1e6:.1f}M | embedding-hariç {non_emb/1e6:.1f}M")
