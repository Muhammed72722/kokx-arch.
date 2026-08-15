"""
Aşama 1 — Mimari Probe Eğitimi (Kaggle T4/P100 üzerinde çalıştır)

Kullanım:
    python train_probe.py --config A_mamba_low_attn --tokens 3_000_000_000
    python train_probe.py --config B_mamba_high_attn --tokens 3_000_000_000
    python train_probe.py --config C_gdn_hybrid       --tokens 3_000_000_000

Her config'i AYNI token bütçesiyle koştur, sonra HumanEval pass@k ile karşılaştır
(bkz. README.md -> "Değerlendirme" bölümü).

Kurulum (Kaggle notebook, ilk hücre):
    !pip install --break-system-packages mamba-ssm causal-conv1d flash-linear-attention \
        transformers datasets accelerate einops
"""

import argparse
import math
import time

import torch
from torch.utils.data import IterableDataset, DataLoader

from hybrid_model import KOKXHybrid
from configs import PROBE_CONFIGS


class PackedCodeStream(IterableDataset):
    """HF datasets üzerinden kod verisini stream edip seq_len'e paketler.

    Varsayılan: bigcode/starcoderdata (gated değil, script değil -- parquet,
    dil bazlı data_dir ile yükleniyor). GitHub issues/commits/jupyter alt
    kümeleri farklı şemaya sahip, bu yüzden data_dir mutlaka bir programlama
    dili klasörü olmalı (python, java, javascript, cpp, vb.), boş bırakma."""

    def __init__(self, tokenizer, seq_len=2048, dataset_name="bigcode/starcoderdata",
                 dataset_config=None, data_dir="python", text_field="content"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.data_dir = data_dir
        self.text_field = text_field

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset(
            self.dataset_name, self.dataset_config,
            data_dir=self.data_dir, split="train", streaming=True,
        )
        buffer = []
        for example in ds:
            text = example.get(self.text_field, "")
            if not text:
                continue
            ids = self.tokenizer(text, truncation=False)["input_ids"]
            buffer.extend(ids + [self.tokenizer.eos_token_id])
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


def get_lr(step, warmup_steps, total_steps, max_lr, min_lr=1e-5):
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=list(PROBE_CONFIGS.keys()))
    parser.add_argument("--tokens", type=int, default=3_000_000_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--dataset_name", type=str, default="bigcode/starcoderdata")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="python")
    parser.add_argument("--text_field", type=str, default="content")
    parser.add_argument("--tokenizer_name", type=str, default="bigcode/starcoder2-3b")
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=2000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = PROBE_CONFIGS[args.config]
    cfg.vocab_size = len(tokenizer)
    cfg.max_seq_len = args.seq_len

    model = KOKXHybrid(cfg).to(device)
    n_params = model.num_params(non_embedding=True)
    print(f"[{args.config}] non-embedding parametre: {n_params / 1e6:.1f}M")
    print(f"[{args.config}] layer_pattern: {cfg.layer_pattern}")

    total_steps = args.tokens // (args.batch_size * args.grad_accum * args.seq_len)
    print(f"Hedef adım sayısı: {total_steps} (token bütçesi: {args.tokens/1e9:.2f}B)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler()

    stream = PackedCodeStream(
        tokenizer, seq_len=args.seq_len,
        dataset_name=args.dataset_name, dataset_config=args.dataset_config,
        data_dir=args.data_dir, text_field=args.text_field,
    )
    loader = DataLoader(stream, batch_size=args.batch_size, num_workers=2)

    step = 0
    t0 = time.time()
    optimizer.zero_grad()
    accum_loss = 0.0

    for micro_step, (x, y) in enumerate(loader):
        if step >= total_steps:
            break
        x, y = x.to(device), y.to(device)
        lr = get_lr(step, args.warmup_steps, total_steps, args.max_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(x, labels=y)
            loss = out["loss"] / args.grad_accum

        scaler.scale(loss).backward()
        accum_loss += loss.item()

        if (micro_step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                tok_per_sec = (step * args.batch_size * args.grad_accum * args.seq_len) / max(elapsed, 1e-5)
                print(f"step {step}/{total_steps} | loss {accum_loss:.4f} | lr {lr:.2e} "
                      f"| {tok_per_sec/1e3:.1f}K tok/s")
            accum_loss = 0.0

            if step % args.save_every == 0:
                path = f"{args.out_dir}/{args.config}_step{step}.pt"
                torch.save({"model": model.state_dict(), "config": cfg, "step": step}, path)
                print(f"kaydedildi -> {path}")

    final_path = f"{args.out_dir}/{args.config}_final.pt"
    torch.save({"model": model.state_dict(), "config": cfg, "step": step}, final_path)
    print(f"Probe tamamlandı -> {final_path}")


if __name__ == "__main__":
    main()
