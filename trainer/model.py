"""TinyGPT model definition.

Same architecture as v1 (nn.TransformerEncoder with a causal mask, norm_first,
GELU) plus batched-example support via key_padding_mask. The parameter names
are unchanged, so v1 checkpoints still load.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class TinyGPT(nn.Module):
    """Small character/token-level GPT using causal Transformer attention."""

    def __init__(
        self, vocab_size, block_size=128, n_embd=128, n_heads=4, n_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        if n_embd % n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")
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

    def forward(self, idx, targets=None, target_mask=None, key_padding_mask=None):
        _, length = idx.shape
        if length > self.block_size:
            raise ValueError(
                f"Input length {length} exceeds block size {self.block_size}."
            )
        positions = torch.arange(length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=idx.device),
            diagonal=1,
        )
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
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
