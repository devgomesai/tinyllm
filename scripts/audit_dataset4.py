"""Inspect short answers, self-referential answers, and sample facts."""

import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "data.txt"
text = DATA.read_text(encoding="utf-8")
examples = [b.strip("\n") for b in text.split("<END>") if b.strip()]
pairs = []
for ex in examples:
    lines = [ln for ln in ex.split("\n") if ln.strip()]
    if len(lines) == 2:
        pairs.append((lines[0][6:].strip(), lines[1][9:].strip()))

print("=== Answers < 20 chars ===")
for q, a in pairs:
    if len(a) < 20:
        print(f"  {q[:60]!r} -> {a!r}")

print("\n=== Answers mentioning TinyLLM/AI ===")
for q, a in pairs:
    if "TinyLLM" in a or "As an AI" in a:
        print(f"  {q[:70]!r} -> {a[:80]!r}")

print("\n=== Random 50 for factual spot-check ===")
random.seed(7)
for q, a in random.sample(pairs, 50):
    print(f"Q: {q}\nA: {a}\n")
