import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "data.txt"
MODEL_FILE = PROJECT_DIR / "model" / "mini_gpt.pt"


class TinyGPT(nn.Module):
    """Small character-level GPT using causal Transformer attention."""

    def __init__(self, vocab_size, block_size=128, n_embd=128, n_heads=4,
                 n_layers=3, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_heads, dim_feedforward=4 * n_embd,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        _, length = idx.shape
        if length > self.block_size:
            raise ValueError(f"Input length {length} exceeds block size {self.block_size}.")
        positions = torch.arange(length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=idx.device), diagonal=1
        )
        x = self.transformer(x, mask=causal_mask)
        logits = self.lm_head(self.final_norm(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def get_batch(data, batch_size, block_size, device):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device):
    model.eval()
    result = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(10):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        result[name] = sum(losses) / len(losses)
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser(description="Train character-level Tiny GPT")
    parser.add_argument("--iters", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

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
    split = int(0.9 * len(encoded))
    train_data, val_data = encoded[:split], encoded[split:]
    if len(val_data) <= args.block_size:
        raise ValueError("Validation text is too short; reduce --block-size.")

    print(f"Device: {device}")
    print(f"Characters in story: {len(text):,}")
    print(f"Vocabulary size: {len(chars)}")
    print(f"Training characters: {len(train_data):,}")
    print(f"Validation characters: {len(val_data):,}")

    config = {"block_size": args.block_size, "n_embd": 128, "n_heads": 4,
              "n_layers": 3, "dropout": 0.1}
    model = TinyGPT(vocab_size=len(chars), **config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.iters + 1):
        if step == 1 or step % args.eval_interval == 0 or step == args.iters:
            losses = estimate_loss(model, train_data, val_data, args.batch_size,
                                   args.block_size, device)
            print(f"Step {step:5d}/{args.iters} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "stoi": stoi, "itos": itos,
                "vocab_size": len(chars), "config": config}, MODEL_FILE)
    print(f"\nModel saved to: {MODEL_FILE}\nTraining complete.")


if __name__ == "__main__":
    main()