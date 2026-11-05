"""Dataset audit for data/data.txt (read-only, prints findings)."""

from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "data.txt"

text = DATA.read_text(encoding="utf-8")
print(f"Total characters: {len(text):,}")
print(f"Distinct chars: {len(set(text))}")
print("Char classes:", sorted(set(text)))

# Parse examples: blocks separated by <END>
examples = []
for block in text.split("<END>"):
    block = block.strip("\n")
    if not block.strip():
        continue
    examples.append(block)

print(f"\nBlocks ending before <END>: {len(examples):,}")

# Parse each example into (question, answer)
UserTag, BotTag = "User:", "TinyLLM:"
ok, bad = [], Counter()
for ex in examples:
    lines = [ln for ln in ex.split("\n") if ln.strip()]
    if len(lines) != 2:
        bad[f"{len(lines)} lines"] += 1
        continue
    q, a = lines
    if not q.startswith(UserTag):
        bad["no User: prefix"] += 1
        continue
    if not a.startswith(BotTag):
        bad["no TinyLLM: prefix"] += 1
        continue
    question = q[len(UserTag):].strip()
    answer = a[len(BotTag):].strip()
    if not question:
        bad["empty question"] += 1
        continue
    if not answer:
        bad["empty answer"] += 1
        continue
    ok.append((question, answer))

print(f"Well-formed pairs: {len(ok):,}")
print("Malformed:", dict(bad))

# Exact duplicates
pair_counts = Counter(ok)
dups = {k: v for k, v in pair_counts.items() if v > 1}
print(f"Unique pairs: {len(pair_counts):,}")
print(f"Pairs appearing >1x: {len(dups):,} (total extra copies: {sum(v - 1 for v in dups.values()):,})")
print("Top duplicated pairs:")
for (q, a), n in pair_counts.most_common(10):
    if n > 1:
        print(f"  x{n}: {q[:60]!r} -> {a[:60]!r}")

# Near-duplicate questions (normalized whitespace/lowercase)
norm = lambda s: " ".join(s.lower().split())
q_counts = Counter(norm(q) for q, _ in ok)
dupq = {q: n for q, n in q_counts.items() if n > 1}
print(f"\nUnique normalized questions: {len(q_counts):,}")
print(f"Normalized questions repeated: {len(dupq):,}")
print("Top repeated questions:")
for q, n in q_counts.most_common(15):
    if n > 1:
        print(f"  x{n}: {q[:80]!r}")

# Questions differing by one word from another (crude): group by sorted word set minus stopwords
stop = set("what is a an the of do does how why in to for on with and".split())
sig_counts = Counter(frozenset(w for w in norm(q).split() if w not in stop) for q, _ in ok)
near = {s: n for s, n in sig_counts.items() if n > 2}
print(f"\nContent-signature groups with >2 variants: {len(near)}")

# Answers quality signals
ans_lens = Counter()
short_ans = 0
self_ref = 0
for q, a in ok:
    na = len(a)
    if na < 20:
        short_ans += 1
    if "TinyLLM" in a or "As an AI" in a:
        self_ref += 1
print(f"\nAnswers shorter than 20 chars: {short_ans}")
print(f"Answers containing 'TinyLLM' or 'As an AI': {self_ref}")

# Length distribution of full examples
full_lens = [len(q) + len(a) for q, a in ok]
import statistics
print(f"Example length (q+a) chars: min={min(full_lens)}, median={statistics.median(full_lens)}, p95={sorted(full_lens)[int(0.95*len(full_lens))]}, max={max(full_lens)}")
over = sum(1 for l in full_lens if l > 110)
print(f"Examples (q+a) longer than 110 chars (context limit with header/marker): {over}")

# Categories heuristic
cats = Counter()
for q, a in ok:
    nq = norm(q)
    if any(k in nq for k in ("+", "-", "times", "divide", "add", "sum of")):
        cats["arithmetic"] += 1
    elif any(k in nq for k in ("python", "def ", "list ", "tuple", "dict", "loop", "elif", "string")):
        cats["python"] += 1
    elif any(k in nq for k in ("neural", "tensor", "gradient", "activation", "cnn", "conv", "pool", "transformer", "attention", "embedding", "token", "backprop", "overfit", "dropout", "batch", "epoch", "loss")):
        cats["ml/nn"] += 1
    else:
        cats["general/other"] += 1
print("\nHeuristic categories:", dict(cats))
