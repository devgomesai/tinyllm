"""TinyGPT 2.0 training pipeline.

Improvements over the v1 trainer:
- Example-level batching (no windows starting mid-example), padded with
  attention masks; loss is masked to answer tokens only.
- Configurable tokenizer: char (v1 style) or a BPE trained on the corpus.
- Proper train/val/test splits from data/processed (see trainer/dataset.py).
- Deterministic seeding, full-dataset validation loss, best+final checkpoints,
  CSV metric logging, graceful OOM handling with batch-size fallback.

Usage:
    uv run python -m trainer.dataset --seed 42          # build splits once
    uv run python trainer/train.py --tokenizer char --iters 8000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from trainer.dataset import (  # noqa: E402
    BOT_TAG,
    END_TAG,
    Example,
    parse_pairs,
)
from trainer.model import TinyGPT  # noqa: E402
from trainer.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer  # noqa: E402

# Kept for compatibility with the v1 inference interface. The preserved
# baseline checkpoints live in model/v1/ (mini_gpt_v2.pt = v1 recipe on
# data/data.txt, mini_gpt_v1.pt = exact data_v1.txt checkpoint).
MODEL_FILE = PROJECT_DIR / "model" / "v1" / "mini_gpt_v2.pt"

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def load_examples(path: Path) -> list[Example]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first with: uv run python -m trainer.dataset"
        )
    examples, stats = parse_pairs(path.read_text(encoding="utf-8"), source=path.name)
    if not examples:
        raise ValueError(f"{path} contains no parseable examples.")
    bad = sum(v for k, v in stats.items() if k != "parsed" and isinstance(v, int))
    if bad:
        print(f"WARNING: {path.name}: {bad} malformed blocks skipped ({stats})")
    return examples


def encode_examples(examples, tokenizer):
    """Encode every example; returns list of (ids list, answer mask list)."""
    encoded = []
    skipped = 0
    for ex in examples:
        prompt_part = f"User: {ex.question}\n{BOT_TAG}"
        answer_part = f" {ex.answer}\n{END_TAG}"
        ids, mask = tokenizer.encode_answer_parts(prompt_part, answer_part)
        if len(ids) < 3:
            skipped += 1
            continue
        encoded.append((ids, mask))
    if skipped:
        print(f"WARNING: skipped {skipped} degenerate examples.")
    return encoded


def make_batch(encoded, indices, device, pad_id=0, max_len=None):
    """Padded batch: (x, y, loss_mask, key_padding_mask)."""
    seqs = [encoded[i] for i in indices]
    lengths = [len(ids) for ids, _ in seqs]
    if max_len is None:
        max_len = max(lengths)
    x = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    y = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    m = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    pad_mask = torch.ones((len(seqs), max_len), dtype=torch.bool)
    for row, ((ids, mask), length) in enumerate(zip(seqs, lengths)):
        x[row, :length] = torch.tensor(ids, dtype=torch.long)
        y[row, : length - 1] = torch.tensor(ids[1:], dtype=torch.long)
        m[row, : length - 1] = torch.tensor(mask[1:], dtype=torch.bool)
        pad_mask[row, :length] = False
    return (
        x.to(device),
        y.to(device),
        m.to(device),
        pad_mask.to(device),
    )


@torch.no_grad()
def evaluate_loss(model, encoded, device, batch_size, pad_id=0):
    """Deterministic loss over an entire encoded dataset, in fixed order."""
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(encoded), batch_size):
        idx = range(start, min(start + batch_size, len(encoded)))
        x, y, m, pads = make_batch(encoded, idx, device, pad_id=pad_id)
        if m.sum() == 0:
            continue
        _, loss = model(x, y, m, key_padding_mask=pads)
        total += loss.item() * m.sum().item()
        count += m.sum().item()
    model.train()
    return total / max(count, 1)


def bpe_from_text(texts, n_merges):
    return BPETokenizer.build("\n\n".join(texts), n_merges=n_merges)


def main():
    parser = argparse.ArgumentParser(
        description="Train TinyGPT 2.0 (char or BPE tokenizer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--bpe-merges", type=int, default=400)
    parser.add_argument("--iters", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr-frac", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-batch", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Checkpoint directory; default model/v2_<tokenizer>")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny smoke run: 30 iters, checkpoint to --out-dir")
    args = parser.parse_args()

    if args.smoke:
        args.iters = min(args.iters, 30)
        args.eval_interval = min(args.eval_interval, 10)
        if args.out_dir is None:
            args.out_dir = PROJECT_DIR / "model" / "smoke"

    out_dir = args.out_dir or PROJECT_DIR / "model" / "v2" / args.tokenizer
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training on CPU (slow).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_file = PROCESSED_DIR / "train.txt"
    val_file = PROCESSED_DIR / "val.txt"
    test_file = PROCESSED_DIR / "test.txt"

    train_examples = load_examples(train_file)
    val_examples = load_examples(val_file)
    test_examples = load_examples(test_file) if test_file.exists() else []
    print(
        f"Splits: {len(train_examples)} train / {len(val_examples)} val "
        f"/ {len(test_examples)} test examples"
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Tokenizer
    if args.tokenizer == "char":
        corpus = "\n\n".join(ex.format() for ex in train_examples)
        tokenizer = CharTokenizer.build(corpus)
        pad_id = 0  # masked out by attention/loss masks; only a filler value
    else:
        corpus = "\n\n".join(ex.format() for ex in train_examples)
        tokenizer = bpe_from_text(
            [ex.format() for ex in train_examples], args.bpe_merges)
        pad_id = tokenizer.pad_id
    print(f"Tokenizer: {tokenizer.name}, vocabulary: {tokenizer.vocab_size}, "
          f"pad id: {pad_id}")

    torch.manual_seed(args.seed)  # reset after tokenizer build for determinism
    train_encoded = encode_examples(train_examples, tokenizer)
    val_encoded = encode_examples(val_examples, tokenizer)
    test_encoded = encode_examples(test_examples, tokenizer)
    max_formatted = max(
        max((len(ids) for ids, _ in train_encoded), default=0),
        max((len(ids) for ids, _ in val_encoded), default=0),
        max((len(ids) for ids, _ in test_encoded), default=0),
    )
    print(f"Longest encoded example: {max_formatted} tokens "
          f"(context {args.block_size})")
    if max_formatted > args.block_size:
        print(
            "WARNING: some examples exceed the context length and will be "
            "CHUNK-EXCLUDED from training. Increase --block-size to use them."
        )
        before = len(train_encoded) + len(val_encoded) + len(test_encoded)
        train_encoded = [(i, m) for (i, m) in train_encoded if len(i) <= args.block_size]
        val_encoded = [(i, m) for (i, m) in val_encoded if len(i) <= args.block_size]
        test_encoded = [(i, m) for (i, m) in test_encoded if len(i) <= args.block_size]
        after = len(train_encoded) + len(val_encoded) + len(test_encoded)
        print(f"Dropped {before - after} over-length examples (not silently truncated).")

    config = {
        "block_size": args.block_size,
        "n_embd": args.n_embd,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
    }
    model = TinyGPT(vocab_size=tokenizer.vocab_size, **config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val = float("inf")
    log_path = out_dir / "training_log.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["step", "train_loss", "val_loss", "lr", "elapsed_s", "oom_events"])

    def save_checkpoint(path: Path, val_loss, step, final):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab_size": tokenizer.vocab_size,
                "config": config,
                "tokenizer": tokenizer.state(),
                "stoi": getattr(tokenizer, "stoi", None),
                "itos": getattr(tokenizer, "itos", None),
                "meta": {
                    "step": step,
                    "val_loss": val_loss,
                    "final": final,
                    "seed": args.seed,
                    "train_examples": len(train_examples),
                    "val_examples": len(val_examples),
                    "test_examples": len(test_examples),
                    "created": datetime.now().isoformat(timespec="seconds"),
                },
            },
            path,
        )

    rng = torch.Generator().manual_seed(args.seed + 1)
    n_train = len(train_encoded)
    batch = args.batch_size
    oom_events = 0
    t0 = time.time()
    warmup = args.warmup_iters

    for step in range(1, args.iters + 1):
        # cosine schedule with warmup (computed first so logging can use it)
        if step <= warmup:
            lr_current = args.lr * step / max(1, warmup)
        else:
            progress = (step - warmup) / max(1, args.iters - warmup)
            cos = 0.5 * (1 + math.cos(math.pi * progress))
            lr_current = args.lr * (
                args.min_lr_frac + (1 - args.min_lr_frac) * cos
            )
        for group in optimizer.param_groups:
            group["lr"] = lr_current

        evaluate_now = (
            step == 1 or step % args.eval_interval == 0 or step == args.iters
        )
        if evaluate_now:
            train_loss = evaluate_loss(
                model, train_encoded, device, args.evaluate_batch, pad_id)
            val_loss = evaluate_loss(
                model, val_encoded, device, args.evaluate_batch, pad_id)
            print(
                f"Step {step:5d}/{args.iters} | train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f}"
            )
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                save_checkpoint(out_dir / "best.pt", val_loss, step, final=False)
                print(f"  best checkpoint saved (val {val_loss:.4f})")
            # Recreate the output folder + log header if the directory was
            # deleted (e.g. artifacts moved while the run was in progress).
            out_dir.mkdir(parents=True, exist_ok=True)
            new_log = not log_path.exists()
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new_log:
                    writer.writerow(
                        ["step", "train_loss", "val_loss", "lr", "elapsed_s",
                         "oom_events"])
                writer.writerow(
                    [step, f"{train_loss:.4f}", f"{val_loss:.4f}",
                     f"{lr_current:.6f}", f"{time.time() - t0:.1f}",
                     oom_events])

        indices = torch.randint(n_train, (batch,), generator=rng).tolist()
        try:
            x, y, m, pads = make_batch(train_encoded, indices, device, pad_id=pad_id)
            _, loss = model(x, y, m, key_padding_mask=pads)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            oom_events += 1
            if batch > 1:
                batch = max(1, batch // 2)
                print(f"CUDA OOM at step {step}: reducing batch size to {batch}")
            else:
                print(f"CUDA OOM at step {step} with batch size 1. Skipping step.")
            del e

    save_checkpoint(out_dir / "final.pt", best_val, args.iters, final=True)
    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3 if str(device).startswith("cuda") else 0.0
    summary = {
        "out_dir": str(out_dir),
        "tokenizer": tokenizer.name,
        "vocab_size": tokenizer.vocab_size,
        "n_params": n_params,
        "best_val_loss": best_val,
        "iters": args.iters,
        "batch_size_final": batch,
        "oom_events": oom_events,
        "elapsed_s": round(time.time() - t0, 1),
        "peak_gpu_memory_gb": round(peak_mem, 3),
        "config": config,
        "args": vars(args),
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nBest validation loss: {best_val:.4f}")
    print(f"Peak GPU memory: {peak_mem:.2f} GB | OOM events: {oom_events}")
    print(f"Checkpoints saved in: {out_dir} (best.pt, final.pt)")
    print("Training complete.")


if __name__ == "__main__":
    main()
