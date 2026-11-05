"""Quantify base-question clustering after stripping wrapper prefixes."""

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "data.txt"
text = DATA.read_text(encoding="utf-8")
examples = [b.strip("\n") for b in text.split("<END>") if b.strip()]
pairs = []
for ex in examples:
    lines = [ln for ln in ex.split("\n") if ln.strip()]
    if len(lines) == 2:
        pairs.append((lines[0][6:].strip(), lines[1][9:].strip()))

WRAPPERS = [
    "please tell me, ",
    "i was wondering, ",
    "can i ask, ",
    "please tell me ",
    "i was wondering ",
    "can i ask ",
    "what do you know about ",
    "what's ",
    "what is ",
    "tell me about ",
    "can you explain ",
    "explain ",
    "how do ",
    "what does ",
    "why do ",
    "how many ",
    "what was ",
    "who was ",
    "who is ",
    "what are ",
    "can you ",
    "is it possible for you ",
    "define ",
    "describe ",
]

def base_q(q):
    n = " ".join(q.lower().replace("?", "").replace(".", "").split())
    changed = True
    while changed:
        changed = False
        for w in WRAPPERS:
            if n.startswith(w):
                n = n[len(w):].strip(" ?.,")
                changed = True
    return n

groups = defaultdict(list)
for q, a in pairs:
    groups[base_q(q)].append((q, a))

sizes = Counter(len(v) for v in groups.values())
print(f"Base-question groups: {len(groups)}")
print(f"Group size histogram: {dict(sorted(sizes.items()))}")
multi = sum(1 for v in groups.values() if len(v) > 1)
print(f"Groups with >1 example: {multi}")

big = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:25]
for k, v in big:
    print(f"\n[{len(v)}x] base: {k[:60]!r}")
    for q, a in v[:3]:
        print(f"    {q[:70]!r} -> {a[:60]!r}")

# Answer-level near-duplicates: same normalized answer with different questions
def norm_a(a):
    return " ".join(a.lower().split())
a_counts = Counter(norm_a(a) for _, a in pairs)
dup_a = {a: n for a, n in a_counts.items() if n > 3}
print(f"\n\nAnswers repeated >3 times: {len(dup_a)}")
for a, n in a_counts.most_common(12):
    if n > 3:
        print(f"  x{n}: {a[:80]!r}")
