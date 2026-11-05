"""Dataset model: parse, clean, group and split TinyLLM dialogue examples.

The canonical unit is one question/answer pair ("example"). Examples are
formatted for training as:

    User: {question}
    TinyLLM: {answer}
    <END>

and files join examples with blank lines ("\n\n"), matching the existing
data/data.txt format. The multi-turn conversation is expressed by making the
question of a later pair reference earlier turns ("And what about...?"), so
single-turn parsing keeps all complexity out of the trainer.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

USER_TAG = "User:"
BOT_TAG = "TinyLLM:"
END_TAG = "<END>"
SEP = "\n\n"

# Wrapper prefixes used across data.txt and by the expansion pipeline.
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
    "how does ",
    "what does ",
    "why do ",
    "how many ",
    "how much ",
    "what was ",
    "who was ",
    "who is ",
    "what are ",
    "can you ",
    "could you ",
    "is it possible for you ",
    "define ",
    "describe ",
    "tell me ",
    "if i have ",
]

# Regex for programmatic arithmetic questions: every integer in the question.
NUM_RE = re.compile(r"\d+")


def normalize_question(q: str) -> str:
    return " ".join(q.lower().replace("?", "").replace(".", "").split())


def base_question(q: str) -> str:
    """Strip wrapper phraes to recover the underlying fact key."""
    n = normalize_question(q)
    changed = True
    while changed:
        changed = False
        for w in WRAPPERS:
            if n.startswith(w):
                n = n[len(w):].strip(" ?.,")
                changed = True
    return n


@dataclass
class Example:
    question: str
    answer: str
    source: str = "data.txt"          # data.txt or expansion/<name>
    group: str = ""                    # fact group key used for splitting
    category: str = "general"          # coarse category tag
    split: str = ""                    # train / val / test, assigned later

    def format(self) -> str:
        return f"{USER_TAG} {self.question}\n{BOT_TAG} {self.answer}\n{END_TAG}"

    def formatted_length(self) -> int:
        return len(self.format())


def parse_pairs(text: str, source: str = "data.txt") -> tuple[list[Example], dict]:
    """Parse text split into <END>-terminated blocks into Examples."""
    examples: list[Example] = []
    stats = Counter(source=source)
    for block in text.split(END_TAG):
        block = block.strip("\n")
        if not block.strip():
            stats["empty_block"] += 1
            continue
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if len(lines) != 2:
            stats["malformed_block"] += 1
            continue
        q_line, a_line = lines
        if not q_line.startswith(USER_TAG):
            stats["bad_user_tag"] += 1
            continue
        if not a_line.startswith(BOT_TAG):
            stats["bad_bot_tag"] += 1
            continue
        question = q_line[len(USER_TAG):].strip()
        answer = a_line[len(BOT_TAG):].strip()
        if not question or not answer:
            stats["empty_field"] += 1
            continue
        examples.append(Example(question, answer, source=source))
        stats["parsed"] += 1
    return examples, dict(stats)


BINARY_OPS = (" + ", " - ", " times ", "multiplied by", "divided by")


def is_arithmetic(question: str) -> bool:
    """True for simple number questions (binary ops, squares, halves, sequences)."""
    n = " ".join(question.lower().split())
    if "next number" in n or "even or odd" in n:
        return True
    if not any(op in n for op in BINARY_OPS):
        return False
    return len(NUM_RE.findall(n)) >= 2 and not any(w in n for w in (
        "digital", "bigger screen", "battery"))


def is_binary_arith(question: str) -> tuple[bool, list[int], str]:
    """Detect two-operand one-op arithmetic; returns (match, operands, normalized)."""
    n = " ".join(question.lower().split())
    matched = [op for op in BINARY_OPS if op in n]
    if not matched:
        return False, [], n
    nums = [int(x) for x in NUM_RE.findall(n)]
    if len(nums) == 2 and not any(w in n for w in (
            "digital", "bigger screen", "battery", "percent")):
        return True, nums, n
    return False, [], n


def arith_band(question: str) -> str:
    """'large' if both operands >= 10, else 'medium'.

    Large-band combinations are held out of training entirely so the test set
    can measure arithmetic on operand combinations the model has never seen.
    """
    ok, nums, _ = is_binary_arith(question)
    if not ok or "percent" in question.lower():
        return "medium"
    if all(v >= 10 for v in nums):
        return "large"
    return "medium"


def reserve_group_key(ex: Example) -> str:
    """Choose the split-group key for an example."""
    if ex.category == "arithmetic" and arith_band(ex.question) == "large":
        return "__arith_large__"        # large-band arithmetic reserved for test
    return "g:" + base_question(ex.question)


def load_raw_examples(data_file: Path | None = None) -> list[Example]:
    data_file = data_file or PROJECT_DIR / "data" / "data.txt"
    text = data_file.read_text(encoding="utf-8")
    examples, _ = parse_pairs(text, source=data_file.name)
    for ex in examples:
        ex.category = categorize(ex.question, ex.answer)
    return examples


KEYWORD_CATEGORIES = [
    ("python", ("python", "def ", "tuple", "list comprehension", "dictionary",
                "for loop", "while loop", "elif", "pylint", "flake8", "pandas",
                "numpy", "matplotlib", "print function", "string method", "slicing")),
    ("torch", ("pytorch", "torch", "tensor", "autograd", "requires_grad",
               "cuda", "dataloader", "optim ", "adam", "linear layer", "nn.",
               "lightning")),
    ("nn", ("neural", "gradient", "activation", "relu", "sigmoid", "softmax",
            "cnn", "conv", "pooling", "batch", "epoch", "overfit", "underfit",
            "dropout", "backprop", "epoch", "weight", "loss function",
            "cross-entropy", "embedding", "transformer", "attention", "head",
            "token", "encoder", "decoder", "hidden layer", "feed-forward",
            "lr", "learning rate", "inference", "gpu", "vram", "checkpoint",
            "fine-tune", "prompt", "inference", "generative")),
    ("arithmetic", ("+", "-", "times", "divide", "divided", "percent",
                    "square", "half of", "even or odd", "double", "next number",
                    "bigger", "smaller")),
]
GENERAL_NAMES = ("hi", "hey", "hello", "thanks", "thank you", "sorry", "see you",
                 "your name", "who are you", "who built", "creator", "version",
                 "goodbye", "birth", "feel", "dream", "sleep", "eat")


def categorize(question: str, answer: str) -> str:
    n = normalize_question(question)
    na = normalize_question(answer)
    if is_arithmetic(n):
        return "arithmetic"
    for cat, words in KEYWORD_CATEGORIES:
        if any(w in n for w in words):
            return cat
    if any(w in n for w in GENERAL_NAMES):
        return "general"
    if "tinyllm" in n or "tinyllm" in na:
        return "identity"
    return "general"


def dedupe(examples: list[Example]) -> tuple[list[Example], dict]:
    """Drop exact (q, a) duplicates and identical normalized questions."""
    stats = Counter()
    seen_pair = set()
    seen_qnorm = set()
    kept: list[Example] = []
    for ex in examples:
        key = (normalize_question(ex.question), " ".join(ex.answer.lower().split()))
        if key in seen_pair:
            stats["exact_duplicates_dropped"] += 1
            continue
        qn = normalize_question(ex.question)
        if qn in seen_qnorm:
            stats["duplicate_questions_dropped"] += 1
            continue
        seen_pair.add(key)
        seen_qnorm.add(qn)
        kept.append(ex)
        stats["kept"] += 1
    return kept, dict(stats)


def group_examples(examples: list[Example]) -> dict[str, list[Example]]:
    """All paraphrase variants of one fact share a split group."""
    groups: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        ex.group = reserve_group_key(ex)
        groups[ex.group].append(ex)
    return groups


def stratified_split(groups: dict[str, list[Example]],
                     ratios=(0.8, 0.1, 0.1), seed: int = 42) -> None:
    """Assign every example's split deterministically, per category stratum.

    Groups incidentally assigned '__arith_large__' fall into test; their split
    is enforced after the fact.
    """
    import random

    rng = random.Random(seed)
    by_cat_group: dict[str, list[str]] = defaultdict(list)
    for gkey, exs in groups.items():
        by_cat_group[exs[0].category].append(gkey)

    for cat, gkeys in by_cat_group.items():
        rng.shuffle(gkeys)
        total = sum(len(groups[g]) for g in gkeys)
        cumulative = 0
        for gkey in gkeys:
            proportion = cumulative / total
            if proportion < ratios[0]:
                split = "train"
            elif proportion < ratios[0] + ratios[1]:
                split = "val"
            else:
                split = "test"
            cumulative += len(groups[gkey])
            for ex in groups[gkey]:
                ex.split = split

    # Enforce arithmetic large-band = test (never in train / val).
    for exs in groups.values():
        if exs[0].group.startswith("__arith_large__") and exs[0].category == "arithmetic":
            for ex in exs:
                ex.split = "test"


def split_stats(examples: list[Example]) -> dict:
    per_split = defaultdict(lambda: {"examples": 0, "chars": 0})
    for ex in examples:
        per_split[ex.split]["examples"] += 1
        per_split[ex.split]["chars"] += ex.formatted_length()
    return {k: dict(v) for k, v in per_split.items()}


def write_splits(examples: list[Example], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        exs = [e for e in examples if e.split == name]
        path = out_dir / f"{name}.txt"
        text = SEP.join(e.format() for e in exs)
        path.write_text(text, encoding="utf-8")
    stats = {
        name: {
            "examples": sum(1 for e in examples if e.split == name),
            "chars": sum(e.formatted_length() for e in examples if e.split == name),
            "groups": len({e.group for e in examples if e.split == name}),
            "by_category": dict(Counter(e.category for e in examples if e.split == name)),
        }
        for name in ("train", "val", "test")
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def build_dataset(seed: int = 42,
                  data_file: Path | None = None,
                  out_dir: Path | None = None) -> dict:
    """Full reproducible pipeline: load, expand, categorize, dedupe, split, write."""
    raw = load_raw_examples(data_file)
    raw_stats = {"raw_examples": len(raw)}

    from trainer.data_expansion import generate_expanded_examples
    expanded = generate_expanded_examples(seed=seed)
    raw_stats["expanded_examples"] = len(expanded)

    cleaned, dedupe_stats = dedupe(raw + expanded)
    groups = group_examples(cleaned)
    stratified_split(groups, seed=seed)
    out_dir = out_dir or PROJECT_DIR / "data" / "processed"
    stats = write_splits(cleaned, out_dir)
    stats.update(raw_stats)
    stats["dedupe"] = dedupe_stats
    stats["unique_groups"] = len(groups)
    (out_dir / "pipeline_report.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build processed dataset splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    s = build_dataset(seed=args.seed)
    print(json.dumps(s, indent=2)[:3000])
