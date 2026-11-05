"""Evaluation framework for TinyGPT.

Measures model behavior on held-out questions across five categories:

  A. memorization   questions sampled from the training split
  B. rewording      held-out questions re-worded away from every trained form
  C. unseen topics  probes of concepts absent from the training corpus
  D. reasoning      unseen number sequences, program output, number comparison
  E. arithmetic     unseen operand combinations, separated by difficulty

Generation metrics (greedy by default, so results are reproducible):
  completion_rate   non-empty answers
  end_rate          answers terminate at <END> (status 'end')
  repetition_rate   answers containing long duplicated substrings
  malformed_rate    answers containing stray 'User:'/'TinyLLM:' text or empty
  mean_char_f1      lexical overlap with the reference answer (A and B)

Reasoning/Arithmetic suites report exact-match accuracy by parsed number.
Loss metrics: answer-masked loss on train / val / test splits. Test loss is
reporting only; validation loss is used for checkpoint selection during
training, never the test loss.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from trainer.dataset import BOT_TAG, END_TAG, arith_band, categorize, is_arithmetic, parse_pairs
from trainer.generate import generate
from trainer.model import TinyGPT
from trainer.tokenizer import load_tokenizer

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "eval_results"

NUM_RE = re.compile(r"\d+")


# ---------------------------------------------------------------------------
# Checkpoint loading (v1 char checkpoints + v2 char/BPE checkpoints)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(ckpt_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "tokenizer" in ckpt and ckpt["tokenizer"]:
        tokenizer = load_tokenizer(ckpt["tokenizer"])
    elif ckpt.get("stoi") is not None:
        tokenizer = load_tokenizer(
            {"type": "char", "stoi": ckpt["stoi"], "itos": ckpt["itos"]})
    else:
        raise ValueError("Checkpoint has no vocabulary.")
    model = TinyGPT(vocab_size=ckpt["vocab_size"], **ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, tokenizer, device


def chars_per_token_estimate(tokenizer) -> float:
    probe = "The quick brown fox 123"
    ids = tokenizer.encode(probe)
    return max(1.0, len(probe) / max(len(ids), 1))


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def encode_one(tokenizer, question: str, answer: str, block_size: int):
    prompt_part = f"User: {question}\n{BOT_TAG}"
    answer_part = f" {answer}\n{END_TAG}"
    ids, mask = tokenizer.encode_answer_parts(prompt_part, answer_part)
    if len(ids) > block_size or len(ids) < 3:
        return None
    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0)
    m = torch.tensor(mask[1:], dtype=torch.bool).unsqueeze(0)
    return x, y, m


@torch.no_grad()
def answer_loss_on_examples(model, tokenizer, examples):
    """Mean answer-masked NLL over examples that fit the context window.
    Over-length examples are counted, never silently truncated."""
    device = next(model.parameters()).device
    model.eval()
    losses = []
    skipped = 0
    for ex in examples:
        enc = encode_one(tokenizer, ex.question, ex.answer, model.block_size)
        if enc is None:
            skipped += 1
            continue
        x, y, m = (t.to(device) for t in enc)
        _, loss = model(x, y, m,
                        key_padding_mask=torch.zeros_like(x, dtype=torch.bool))
        losses.append(loss.item())
    model.train()
    mean = float(sum(losses) / max(len(losses), 1))
    return mean, len(losses), skipped


def load_split_examples(name: str, limit=None, seed=42, categories=None,
                        arithmetic_band=None):
    path = PROCESSED_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing split file: {path}. Run trainer.dataset first.")
    examples, _ = parse_pairs(path.read_text(encoding="utf-8"), source=name)
    for ex in examples:
        ex.category = categorize(ex.question, ex.answer)
    if categories:
        examples = [e for e in examples if e.category in categories]
    if arithmetic_band is not None:
        examples = [
            e for e in examples
            if is_arithmetic(e.question) and arith_band(e.question) == arithmetic_band
        ]
    if limit and len(examples) > limit:
        rng = random.Random(seed)
        examples = rng.sample(examples, limit)
    return examples


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def normalize_for_match(s: str) -> str:
    return " ".join(s.lower().replace(",", "").replace(".", "").split())


def token_f1(pred: str, ref: str) -> float:
    pred_t = set(normalize_for_match(pred).split())
    ref_t = set(normalize_for_match(ref).split())
    if not pred_t or not ref_t:
        return 0.0
    overlap = len(pred_t & ref_t)
    prec = overlap / len(pred_t)
    rec = overlap / len(ref_t)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def has_long_repeat(text: str) -> bool:
    """Detects duplicated long substrings (copying/loop artifacts)."""
    n = normalize_for_match(text)
    if len(n) < 50:
        return False
    size = 25
    seen = set()
    for i in range(0, len(n) - size + 1, 3):
        chunk = n[i:i + size]
        if chunk in seen:
            return True
        seen.add(chunk)
    return False


def parse_numbers(text: str) -> list[int]:
    return [int(x) for x in NUM_RE.findall(text)]


def run_generation_suite(model, tokenizer, device, items, greedy=True,
                         temperature=0.0, max_new_tokens=None, notes=None):
    """Run generation for each item; aggregate standard metrics.

    Item fields: question, optional reference, optional expected,
    optional stem (topic word to check), optional difficulty.
    Returns (metrics, rows).
    """
    cpt = chars_per_token_estimate(tokenizer)
    if max_new_tokens is None:
        max_new_tokens = min(200, int(190 / (cpt if tokenizer.name == "bpe" else 1)) + 8)
    metrics = {"total": 0, "completion": 0, "ended_end": 0, "ended_marker": 0,
               "ended_max_tokens": 0, "ended_error": 0, "repetitive": 0,
               "malformed": 0,
               "correct": 0, "unparseable": 0, "f1_sum": 0.0, "stem_hit": 0}
    rows = []
    for it in items:
        try:
            text, status, _ = generate(
                model, tokenizer, it["question"],
                torch.device(next(model.parameters()).device),
                max_new_tokens=max_new_tokens,
                temperature=0.0 if greedy else temperature,
                top_k=0 if greedy else 10,
                top_p=0.0,
                repetition_penalty=1.0,
            )
        except ValueError as exc:
            text, status = "", f"error:{type(exc).__name__}"
        m = metrics
        m["total"] += 1
        m["completion"] += int(bool(text.strip()))
        m[f"ended_{status if status in ('end', 'marker', 'max_tokens') else 'error'}"] += 1
        if has_long_repeat(text):
            m["repetitive"] += 1
        low = normalize_for_match(text)
        if ("user:" in low) or ("tinyllm:" in low) or not text.strip():
            m["malformed"] += 1
        if "expected" in it:
            nums = parse_numbers(text)
            if int(it["expected"]) in nums:
                m["correct"] += 1
            elif not nums:
                m["unparseable"] += 1
        if "reference" in it and it["reference"]:
            m["f1_sum"] += token_f1(text, it["reference"])
        if "stem" in it:
            m["stem_hit"] += int(it["stem"] in normalize_for_match(text))
        rows.append({"question": it["question"],
                     "reference": it.get("reference", ""),
                     "generated": text, "status": status})
    n = max(metrics["total"], 1)
    out = {
        "n": metrics["total"],
        "completion_rate": round(metrics["completion"] / n, 3),
        "end_rate": round(metrics["ended_end"] / n, 3),
        "max_tokens_rate": round(metrics["ended_max_tokens"] / n, 3),
        "marker_rate": round(metrics["ended_marker"] / n, 3),
        "error_rate": round(metrics["ended_error"] / n, 3),
        "repetition_rate": round(metrics["repetitive"] / n, 3),
        "malformed_rate": round(metrics["malformed"] / n, 3),
    }
    if any("expected" in it for it in items):
        out["exact_match"] = round(metrics["correct"] / n, 3)
        out["unparseable_rate"] = round(metrics["unparseable"] / n, 3)
    if any("reference" in it and it["reference"] for it in items):
        out["mean_token_f1"] = round(metrics["f1_sum"] / n, 3)
    if any("stem" in it for it in items):
        out["topic_word_rate"] = round(metrics["stem_hit"] / n, 3)
    if notes:
        out["notes"] = notes
    return out, rows


# ---------------------------------------------------------------------------
# Suite builders
# ---------------------------------------------------------------------------

REWORDINGS = [
    lambda q: f"Could you explain: {q.rstrip('?')}?",
    lambda q: f"Simply put, what does this mean: {q.rstrip('?')}",
    lambda q: f"I don't fully get it. {q}",
    lambda q: f"For a beginner, {q.rstrip('?')}?",
    lambda q: f"In one short reply, {q.rstrip('?').lower()}",
    lambda q: f"Help me understand: {q.rstrip('?')}",
]


def build_reworded_items(examples, n, seed=42):
    """Reword held-out questions; a rewording is only used if it does not
    appear anywhere in the raw training corpus or the processed splits."""
    rng = random.Random(seed)
    corpus = (PROJECT_DIR / "data" / "data.txt").read_text(encoding="utf-8")
    processed = "".join(p.read_text(encoding="utf-8") for p in PROCESSED_DIR.glob("*.txt"))
    bank = corpus + processed
    items = []
    pool = list(examples)
    rng.shuffle(pool)
    for ex in pool:
        if len(items) >= n:
            break
        candidates = [tpl(ex.question) for tpl in REWORDINGS]
        novel = [c for c in candidates if c not in bank]
        if novel:
            items.append({"question": rng.choice(novel), "reference": ex.answer})
    return items


# Unseen-topic probes: each stem verified absent from data.txt + processed splits.
UNSEEN_PROBES = [
    ("What is a capacitor?", "capacitor"),
    ("What is a sundial?", "sundial"),
    ("What is a database transaction?", "transaction"),
    ("What is a lighthouse?", "lighthouse"),
    ("What is a hurricane?", "hurricane"),
    ("What is a compass?", "compass"),
    ("What is a telescope?", "telescope"),
    ("What is a submarine?", "submarine"),
    ("What is a tornado?", "tornado"),
    ("What is insulation in a house?", "insulation"),
]


def build_unseen_items():
    bank = (PROJECT_DIR / "data" / "data.txt").read_text(encoding="utf-8").lower()
    processed = "".join(
        p.read_text(encoding="utf-8").lower() for p in PROCESSED_DIR.glob("*.txt"))
    bank += processed
    leaked = [q for q, stem in UNSEEN_PROBES if stem in bank]
    if leaked:
        raise ValueError(
            f"Unseen-topic probes leak into training data: {leaked}. "
            "Replace them before evaluating.")
    return [{"question": q, "stem": stem} for q, stem in UNSEEN_PROBES]


def build_sequence_items(n, seed=42):
    """Unseen step sizes 13..25; training sequences used steps 1..9."""
    rng = random.Random(seed)
    items = []
    for step in range(13, 26):
        start = rng.randint(4, 40)
        seq = [start]
        for _ in range(3):
            seq.append(seq[-1] + step)
        items.append({
            "question": f"What is the next number: {', '.join(map(str, seq))}?",
            "reference": (f"The next number is {seq[-1] + step}, because each "
                          f"number goes up by {step}."),
            "expected": seq[-1] + step,
            "difficulty": "sequence_step_13_25",
        })
    return items[:n]


def build_program_output_items(n, seed=42):
    items = []
    exprs = [
        ("14 + 27", "41"), ("19 - 8", "11"), ("6 * 7", "42"),
        ("\"dog\" * 3", "dogdogdog"), ("len(\"elephant\")", "8"),
        ("len([4, 8, 15, 16])", "4"), ("max(13, 44)", "44"), ("min(21, 7)", "7"),
        ("sum([5, 10, 15])", "30"), ("\"tiny\".upper()", "TINY"),
        ("[3, 6, 9][-1]", "9"), ("\"flower\".title()", "Flower"),
    ]
    for expr, value in exprs[:n]:
        items.append({
            "question": f"What does print({expr}) show in Python?",
            "reference": f"It prints {value}.",
        })
    return items


def build_comparison_items(n, seed=42):
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        a = rng.randint(20, 99)
        b = rng.randint(100, 999)
        if rng.random() < 0.5:
            a, b = b, a
        items.append({
            "question": f"Which is bigger, {a} or {b}?",
            "reference": f"{max(a, b)} is bigger than {min(a, b)}.",
            "expected": max(a, b),
            "difficulty": "compare_two_numbers",
        })
    return items


def build_arithmetic_items(n, seed=42):
    """Both operands >= 10 -> combinations never present in training.
    Training arithmetic used the medium band (at least one operand < 10)."""
    rng = random.Random(seed)
    tiers = [
        ("small_10_29", lambda: (rng.randint(10, 29), rng.randint(10, 29))),
        ("mid_30_99", lambda: (rng.randint(30, 99), rng.randint(30, 99))),
        ("large_100_plus", lambda: (rng.randint(100, 999), rng.randint(10, 99))),
    ]
    per_tier = max(1, n // 3)
    items = []
    for tier_name, sampler in tiers:
        for _ in range(per_tier):
            a, b = sampler()
            which = rng.randrange(3)
            if which == 0:
                q, expected = f"What is {a} + {b}?", a + b
            elif which == 1:
                if b > a:
                    a, b = b, a
                q, expected = f"What is {a} - {b}?", a - b
            else:
                q, expected = f"What is {a} times {b}?", a * b
            expr = q[len("What is "):-1]
            items.append({
                "question": q,
                "reference": f"{expr} is {expected}.",
                "expected": expected,
                "difficulty": tier_name,
            })
    return items


def number_match_exact(text: str, expected: int) -> bool:
    """The expected number must appear as a whole parsed number somewhere in
    the generated answer (answers never echo the question, so this is safe)."""
    return int(expected) in parse_numbers(text)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(ckpt_path: Path, n_per_suite=40, seed=42,
                        temperature=0.7, sampled_metrics=False,
                        include_loss=True, out_path: Path | None = None,
                        sample_rows=4):
    model, tokenizer, device = load_model_and_tokenizer(ckpt_path)
    greedy = not sampled_metrics
    report = {
        "checkpoint": str(ckpt_path),
        "created": datetime.now().isoformat(timespec="seconds"),
        "tokenizer": tokenizer.name,
        "vocab_size": tokenizer.vocab_size,
        "block_size": model.block_size,
        "device": str(device),
        "seed": seed,
        "greedy_metrics": greedy,
        "params": sum(p.numel() for p in model.parameters()),
    }

    if include_loss:
        for split in ("train", "val", "test"):
            exs = load_split_examples(
                split, limit=250 if split == "train" else None, seed=seed)
            loss, n_used, skipped = answer_loss_on_examples(model, tokenizer, exs)
            report[f"loss_{split}"] = {
                "mean_answer_loss": round(loss, 4),
                "examples_used": n_used,
                "examples_skipped_context": skipped,
            }

    samples: dict[str, list] = {}

    # A. memorization (training questions, greedy)
    train_items = [
        {"question": e.question, "reference": e.answer}
        for e in load_split_examples("train", limit=n_per_suite, seed=seed)
    ]
    suite, rows = run_generation_suite(
        model, tokenizer, device, train_items, greedy=greedy,
        temperature=temperature)
    report["A_memorization"] = suite
    samples["A_memorization"] = rows[:sample_rows]

    # B. reworded held-out questions
    test_exs = load_split_examples("test", seed=seed)
    non_num_test = [e for e in test_exs if not is_arithmetic(e.question)]
    items = build_reworded_items(non_num_test, n=n_per_suite, seed=seed)
    suite, rows = run_generation_suite(
        model, tokenizer, device, items, greedy=greedy, temperature=temperature)
    suite["reworded_count"] = len(items)
    report["B_rewording"] = suite
    samples["B_rewording"] = rows[:sample_rows]

    # C. unseen topics
    items = build_unseen_items()
    suite, rows = run_generation_suite(
        model, tokenizer, device, items, greedy=greedy, temperature=temperature)
    report["C_unseen_topics"] = suite
    samples["C_unseen_topics"] = rows[:sample_rows]

    # D. reasoning
    suite_d = {}
    for name, builder, cnt in (
        ("unseen_sequences", build_sequence_items, max(20, n_per_suite // 2)),
        ("program_output", build_program_output_items, max(12, n_per_suite // 3)),
        ("number_comparison", build_comparison_items, max(12, n_per_suite // 3)),
    ):
        items = builder(cnt, seed=seed)
        s, rows = run_generation_suite(
            model, tokenizer, device, items, greedy=greedy,
            temperature=temperature)
        suite_d[name] = s
        samples[f"D_{name}"] = rows[:sample_rows]
    report["D_reasoning"] = suite_d

    # E. arithmetic: unseen operand combinations, by difficulty tier
    e_report = {}
    arith_items = build_arithmetic_items(n_per_suite, seed=seed)
    for tier in ("small_10_29", "mid_30_99", "large_100_plus"):
        items = [i for i in arith_items if i["difficulty"] == tier]
        s, rows = run_generation_suite(
            model, tokenizer, device, items, greedy=greedy,
            temperature=temperature)
        e_report[tier] = s
        samples[f"E_{tier}"] = rows[:sample_rows]
    e_report["overall"] = {
        "exact_match": round(sum(
            e_report[t]["exact_match"] * e_report[t]["n"]
            for t in ("small_10_29", "mid_30_99", "large_100_plus"))
            / arith_items.__len__(), 3),
        "n": arith_items.__len__(),
    }
    # reserved large-band examples from the original corpus (test split)
    reserved = load_split_examples("test", seed=seed, arithmetic_band="large")
    if reserved:
        items = [
            {"question": e.question, "reference": e.answer,
             "expected": parse_numbers(e.answer)[-1]}
            for e in reserved[:80]
        ]
        s, rows = run_generation_suite(
            model, tokenizer, device, items, greedy=greedy,
            temperature=temperature)
        e_report["reserved_data_txt_large"] = s
        samples["E_reserved"] = rows[:sample_rows]
    report["E_arithmetic"] = e_report

    report["samples"] = samples
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = RESULTS_DIR / (
            f"eval_{ckpt_path.stem}_{tokenizer.name}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_path.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    return report, out_path


def main():
    parser = argparse.ArgumentParser(
        description="Run held-out evaluation for a TinyGPT checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path,
                        default=PROJECT_DIR / "model" / "v1" / "mini_gpt_v2.pt",
                        help="Checkpoint to evaluate")
    parser.add_argument("--n-per-suite", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="sampling temperature for --sampled-metrics")
    parser.add_argument("--sampled-metrics", action="store_true",
                        help="use temperature sampling instead of greedy")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-loss", action="store_true")
    args = parser.parse_args()
    report, out_path = evaluate_checkpoint(
        args.model,
        n_per_suite=args.n_per_suite,
        seed=args.seed,
        temperature=args.temperature,
        sampled_metrics=args.sampled_metrics,
        include_loss=not args.no_loss,
        out_path=args.out,
    )
    print(json.dumps(
        {k: v for k, v in report.items() if not k.startswith("loss_")
         and k != "samples"}, indent=2)[:1200])
    print(f"\nFull report saved to: {out_path}")


if __name__ == "__main__":
    main()
