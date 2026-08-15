# KOK-X — Kod-Odaklı Hibrit Mimari Probe'u

## Ne yapıyoruz

Kod üretiminde hangi sequence-mixer'ın (Mamba-2, Gated DeltaNet, ya da attention oranı
farklı bir hibrit) en iyi sonucu verdiğini **tahmin etmiyoruz, ölçüyoruz.** Araştırma
şunu gösterdi: kod üretimi genel akıl yürütme benchmark'larından çok daha fazla backbone
seçimine duyarlı — yani burada körü körüne "literatürde popüler olan" mimariyi seçmek
yanlış olabilir.

Üç config'i (`configs.py`) **aynı boyut, aynı veri, aynı eğitim tarifiyle** yarıştırıp
kazananı Aşama 2'nin (tam ölçek eğitim) mimarisi olarak kilitleyeceğiz.

## Kurulum (Kaggle notebook, GPU açık)

```bash
pip install --break-system-packages mamba-ssm causal-conv1d flash-linear-attention \
    transformers datasets accelerate einops
```

`mamba-ssm` ve `causal-conv1d` CUDA derlemesi gerektirir — Kaggle'ın T4/P100 imajında
çalışır ama ilk kurulum birkaç dakika sürebilir. `flash-linear-attention` (fla) paketi
Gated DeltaNet dahil çok sayıda linear-attention varyantını içeriyor.

## Parametre sayısını doğrula

```bash
python count_params.py
```

Üç config de aynı `dim=640, n_layers=24` iskeletinde, tek fark layer_pattern
(hangi katman mamba2/gdn/attn). Bu yüzden parametre sayıları birbirine çok yakın
çıkmalı — adil karşılaştırmanın ön koşulu bu.

## Probe'u çalıştır (üç config, sırayla ya da paralel hesaplarda)

```bash
python train_probe.py --config A_mamba_low_attn  --tokens 3000000000
python train_probe.py --config B_mamba_high_attn --tokens 3000000000
python train_probe.py --config C_gdn_hybrid      --tokens 3000000000
```

Varsayılan veri kaynağı `bigcode/the-stack-smol` (küçük, hızlı indirilen kod korpusu).
Asıl KOK-X veri karışımın hazır olunca `--dataset_name` / `--dataset_config` /
`--text_field` ile değiştir.

`--tokens 3_000_000_000` (~3B token) tek bir Kaggle GPU-günü civarında tamamlanacak
şekilde ayarlandı. Bütçen izin veriyorsa 5B'ye çıkarmak sinyali netleştirir.

## Değerlendirme: HumanEval pass@k

Bu repo kasıtlı olarak kod-çalıştırma sandbox'ı içermiyor — üretilen kodu güvenli
izole ortamda çalıştırmak ayrı bir altyapı ister. Bunun yerine, sektör standardı olan
**bigcode-evaluation-harness**'ı kullan:

```bash
git clone https://github.com/bigcode-project/bigcode-evaluation-harness
```

`model.generate()` (`hybrid_model.py` içinde tanımlı, top-p sampling) fonksiyonunu
harness'ın custom-model arayüzüne bağla, `--tasks humaneval` ile pass@1/pass@10 al.
Üç checkpoint'i (`checkpoints/A_..._final.pt`, `B_...`, `C_...`) aynı harness
komutuyla değerlendir, sayıları doğrudan karşılaştır.

## Karar kuralı

- HumanEval pass@1'de en az **+1.5 puan** fark eden config kazanır (gürültü payı için).
- Fark 1.5 puanın altındaysa: throughput/bellek verimliliğine göre karar ver
  (Mamba-2/GDN katmanları attention'dan daha ucuz — eşit performansta daha ucuz olan kazanır).
- Kazanan mimari Aşama 2'nin (250-400M parametre, tam token bütçesi) tarifi olur.

## Sıradaki adım

Probe sonuçları geldiğinde bana ilet — kazanan config'i alıp Aşama 2 için
tokenizer eğitimi + tam veri pipeline'ı + ölçeklendirilmiş config'i birlikte kuralım.
