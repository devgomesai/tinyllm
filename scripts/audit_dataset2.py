"""Deeper audit: near-duplicates, samples, categories."""

from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "data.txt"
text = DATA.read_text(encoding="utf-8")
examples = [b.strip("\n") for b in text.split("<END>") if b.strip()]
pairs = []
for ex in examples:
    lines = [ln for ln in ex.split("\n") if ln.strip()]
    if len(lines) == 2:
        q = lines[0]
        a = lines[1]
        pairs.append((q[len("User:"):].strip(), a[len("TinyLLM:"):].strip()))

def norm(s):
    return " ".join(s.lower().replace("?", " ?").replace(",", "").split())

# Aggressive near-dup: same first 5 normalized words OR high ratio answers for same question shape
first5 = Counter(" ".join(norm(q).split()[:5]) for q, _ in pairs)
print("Groups sharing first 5 words (top 20):")
for k, n in first5.most_common(20):
    if n > 1:
        print(f"  x{n}: {k!r}")

# Sample answers per bucket
print("\n--- Sample Q/A pairs ---")
import random
random.seed(0)
for q, a in random.sample(pairs, 12):
    print(f"Q: {q}\nA: {a}\n")

# Arithmetic subset
arith = [(q, a) for q, a in pairs if any(s in q for s in ("What is", "what is")) and any(c.isdigit() for c in q) ]
print(f"Questions containing digits: {len(arith)}")
for q, a in arith[:30]:
    print(f"  {q!r} -> {a!r}")

# Look for answers containing digits (program outputs etc.)
codish = [(q, a) for q, a in pairs if any(k in norm(q) for k in ("print", "output", "code", "program"))]
print(f"\nProgram/output questions: {len(codish)}")
for q, a in codish[:20]:
    print(f"  {q!r} -> {a!r}")
