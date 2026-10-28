import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "data.txt"
MODEL_FILE = PROJECT_DIR / "model" / "mini_gpt.pt"


class TinyGPT(nn.Module):
    """Small character-level GPT using causal Transformer attention."""

    def __init__(
        self, vocab_size, block_size=128, n_embd=128, n_heads=4, n_layers=3, dropout=0.1
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_heads,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, target_mask=None):
        _, length = idx.shape
        if length > self.block_size:
            raise ValueError(
                f"Input length {length} exceeds block size {self.block_size}."
            )
        positions = torch.arange(length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=idx.device), diagonal=1
        )
        x = self.transformer(x, mask=causal_mask)
        logits = self.lm_head(self.final_norm(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )
        if target_mask is not None:
            m = target_mask.reshape(-1).float()
            loss = (loss * m).sum() / m.sum().clamp(min=1.0)
        else:
            loss = loss.mean()
        return logits, loss


def build_answer_mask(text):
    """True for every character that belongs to an answer (after 'TinyLLM:' up to and including '<END>')."""
    mask = torch.zeros(len(text), dtype=torch.bool)
    pos = 0
    while True:
        start = text.find("TinyLLM:", pos)
        if start == -1:
            break
        start += len("TinyLLM:")
        end = text.find("<END>", start)
        if end == -1:
            break
        end += len("<END>")
        mask[start:end] = True
        pos = end
    return mask


def get_batch(data, mask, batch_size, block_size, device):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    m = torch.stack([mask[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device), m.to(device)


@torch.no_grad()
def estimate_loss(
    model, train_data, val_data, train_mask, val_mask, batch_size, block_size, device
):
    model.eval()
    result = {}
    for name, data, mask in (
        ("train", train_data, train_mask),
        ("val", val_data, val_mask),
    ):
        losses = []
        for _ in range(20):
            x, y, m = get_batch(data, mask, batch_size, block_size, device)
            _, loss = model(x, y, m)
            losses.append(loss.item())
        result[name] = sum(losses) / len(losses)
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser(description="Train character-level Tiny GPT")
    parser.add_argument("--iters", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--keep",
        choices=["last", "best"],
        default="last",
        help="Which weights to save: the final model, or the best validation loss",
    )
    args = parser.parse_args()

    if args.n_embd % args.n_heads != 0:
        parser.error("--n-embd must be divisible by --n-heads")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Training text not found: {DATA_FILE}")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = DATA_FILE.read_text(encoding="utf-8")
    if len(text) < args.block_size + 20:
        raise ValueError("Training text is too short; reduce --block-size.")

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encoded = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    answer_mask = build_answer_mask(text)
    split = int(0.9 * len(encoded))
    train_data, val_data = encoded[:split], encoded[split:]
    train_mask, val_mask = answer_mask[:split], answer_mask[split:]
    if len(val_data) <= args.block_size:
        raise ValueError("Validation text is too short; reduce --block-size.")

    print(f"Device: {device}")
    print(f"Characters in dataset: {len(text):,}")
    print(f"Vocabulary size: {len(chars)}")
    print(f"Training characters: {len(train_data):,}")
    print(f"Validation characters: {len(val_data):,}")
    print(f"Answer characters (loss is computed only on these): {int(answer_mask.sum()):,}")

    config = {
        "block_size": args.block_size,
        "n_embd": args.n_embd,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
    }
    model = TinyGPT(vocab_size=len(chars), **config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val_loss = float("inf")

    def save_checkpoint():
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "vocab_size": len(chars),
                "config": config,
            },
            MODEL_FILE,
        )

    for step in range(1, args.iters + 1):
        should_evaluate = (
            step == 1 or step % args.eval_interval == 0 or step == args.iters
        )

        if should_evaluate:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                train_mask,
                val_mask,
                args.batch_size,
                args.block_size,
                device,
            )
            print(
                f"Step {step:5d}/{args.iters} | "
                f"train answer loss: {losses['train']:.4f} | "
                f"val answer loss: {losses['val']:.4f}"
            )

            val_improved = losses["val"] < best_val_loss
            if val_improved:
                best_val_loss = losses["val"]

            if args.keep == "best" and val_improved:
                save_checkpoint()
                print(f"  Best checkpoint saved (validation loss: {best_val_loss:.4f})")
            elif args.keep == "last" and step == args.iters:
                save_checkpoint()
                print("  Final checkpoint saved")

        lr = max(
            args.lr * 0.5 * (1 + math.cos(math.pi * step / args.iters)), args.lr * 0.1
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y, m = get_batch(
            train_data, train_mask, args.batch_size, args.block_size, device
        )
        _, loss = model(x, y, m)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print(f"\nBest validation answer loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_FILE} (keep={args.keep})")
    print("Training complete.")


if __name__ == "__main__":
    main()