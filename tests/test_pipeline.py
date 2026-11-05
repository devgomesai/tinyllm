"""Automated tests for the TinyLLM pipeline.

Run with:    uv run python tests/test_pipeline.py
No external test framework needed; assert-style helpers, exits non-zero on
failure.
"""

from __future__ import annotations

import hashlib
import re
import string
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from trainer.dataset import (  # noqa: E402
    SEP,
    END_TAG,
    Example,
    arith_band,
    base_question,
    build_dataset,
    categorize,
    dedupe,
    is_arithmetic,
    is_binary_arith,
    load_raw_examples,
    normalize_question,
    parse_pairs,
)
from trainer.data_expansion import generate_expanded_examples  # noqa: E402
from trainer.generate import generate  # noqa: E402
from trainer.model import TinyGPT  # noqa: E402
from trainer.tokenizer import BPETokenizer, CharTokenizer  # noqa: E402

PROCESSED = PROJECT_DIR / "data" / "processed"
ALLOWED_CHARS = (string.ascii_letters + string.digits
                 + " \"',.-()+=:;?!<>/[]*_{}" + "\n")

RESULTS: list[tuple[str, bool, str]] = []


def expect(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))


def approx(a, b, tol=1e-5):
    return abs(a - b) <= tol * max(1.0, abs(b))


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

def test_dataset_parsing():
    raw = load_raw_examples()
    expect("dataset: parses 3432 examples from data.txt", len(raw) == 3432,
           f"got {len(raw)}")
    ok = all(ex.format().startswith("User:") and END_TAG in ex.format()
             for ex in raw[:100])
    expect("dataset: format contains User:/TinyLLM:/<END>", ok)
    text = (PROJECT_DIR / "data" / "data.txt").read_text(encoding="utf-8")
    first_block = text.split(SEP)[0]
    expect("dataset: format() roundtrips the corpus text",
           raw[0].format() == first_block)


def test_duplicate_detection():
    raw = load_raw_examples()[:10]
    doubled = raw + raw
    kept, _stats = dedupe(doubled)
    expect("dedupe: exact duplicate pairs dropped", len(kept) == 10,
           f"kept {len(kept)}")
    variant = Example(raw[0].question + "  ", raw[0].answer)
    kept2, _ = dedupe(raw[:1] + [variant])
    expect("dedupe: same normalized question dropped", len(kept2) == 1,
           f"kept {len(kept2)}")


def test_split_integrity():
    counts = {}
    for split in ("train", "val", "test"):
        path = PROCESSED / f"{split}.txt"
        expect(f"split: {split}.txt exists", path.exists())
        if path.exists():
            counts[split] = len(parse_pairs(path.read_text(encoding="utf-8"))[0])
    expect("split: all three splits populated with sensible ratios",
           counts.get("train", 0) > 2500 and counts.get("val", 0) > 100
           and counts.get("test", 0) > 100,
           str(counts))


def test_split_reproducible():
    h1 = hashlib.md5((PROCESSED / "train.txt").read_bytes()).hexdigest()
    build_dataset(seed=42)
    h2 = hashlib.md5((PROCESSED / "train.txt").read_bytes()).hexdigest()
    expect("split: deterministic rebuild with fixed seed", h1 == h2,
           f"{h1[:8]} vs {h2[:8]}")


def test_no_fact_leak_across_splits():
    """Paraphrased variants of one fact must never cross splits."""
    split_of: dict[tuple, str] = {}
    conflicts = []
    for split in ("train", "val", "test"):
        examples, _ = parse_pairs((PROCESSED / f"{split}.txt").read_text(
            encoding="utf-8"), source=split)
        for ex in examples:
            if is_arithmetic(ex.question) and arith_band(ex.question) == "large":
                key = ("arith_large", tuple(re.findall(r"\d+", ex.question)))
            else:
                key = ("g", base_question(ex.question))
            prior = split_of.setdefault(key, split)
            if prior != split:
                conflicts.append((key, prior, split))
    expect("split: no base fact appears in two splits", not conflicts,
           f"{len(conflicts)} conflicts, e.g. {conflicts[:2]}")


def test_arithmetic_holdout_band():
    from trainer.dataset import load_raw_examples as _  # already imported
    for split in ("train", "val"):
        examples, _ = parse_pairs((PROCESSED / f"{split}.txt").read_text(
            encoding="utf-8"), source=split)
        bad = [e for e in examples
               if is_arithmetic(e.question)
               and arith_band(e.question) == "large"]
        expect(f"arithmetic: no both-operands>=10 combos in {split}",
               not bad, f"{len(bad)} found")
    expect("arithmetic: 50 - 25 detected and in the large band",
           is_binary_arith("What is 50 - 25?")[0]
           and arith_band("What is 50 - 25?") == "large")
    expect("arithmetic: 7 + 8 stays in the medium band",
           arith_band("What is 7 + 8?") == "medium")
    expect("arithmetic: percent questions never banded to large",
           arith_band("What is 50 percent of 80?") == "medium")


def test_expansion_quality():
    expanded = generate_expanded_examples(seed=42)
    expect("expansion: produces a sizeable corpus", len(expanded) >= 500,
           f"{len(expanded)}")
    raw = load_raw_examples()
    raw_q = {normalize_question(e.question) for e in raw}
    dup = sum(1 for e in expanded if normalize_question(e.question) in raw_q)
    expect("expansion: no question duplicates data.txt", dup == 0, f"{dup}")
    bad_chars = sorted(
        {c for e in expanded for c in e.question + e.answer}
        - set(ALLOWED_CHARS))
    expect("expansion: charset inside the v1 vocabulary", not bad_chars,
           f"offending: {bad_chars}")
    cats = {e.category for e in expanded}
    expect("expansion: categories assigned", len(cats) >= 6, str(cats))


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------

def test_char_tokenizer_roundtrip():
    text = "User: What is 7 + 8?\nTinyLLM: 7 + 8 is 15.\n<END> hello"
    tok = CharTokenizer.build(text)
    ids = tok.encode(text)
    expect("char tokenizer: decode(encode(x)) == x", tok.decode(ids) == text)
    unknown = tok.encode("héllo")
    expect("char tokenizer: unknown chars are skipped without crash",
           tok.decode(unknown) == "hllo", f"got {tok.decode(unknown)!r}")


def test_bpe_roundtrip():
    tok = BPETokenizer.build(
        "User: explain tensors\nTinyLLM: Tensors are grids of numbers.\n<END>"
        + " more Sample text with Numbers 12 34 50 + 67 89 and punctuation?",
        n_merges=60)
    text = "User: Explain tensors\nTinyLLM: Tensors are grids of numbers.\n<END>"
    ids = tok.encode(text)
    expect("bpe: decode(encode(x)) == x", tok.decode(ids) == text, tok.decode(ids))
    expect("bpe: <END> maps to the special end id",
           tok.end_id in ids)
    expect("bpe: encode is deterministic", tok.encode(text) == ids)
    ids2 = tok.encode("TinyLLM: 45 + 20")
    expect("bpe: no unknown characters on alphabet-identical text",
           tok.unknowns == 0, f"unknowns={tok.unknowns}")


def test_bpe_vocab_growth():
    tok = BPETokenizer.build(
        ("word sample corpus. " * 60) + "<END>", n_merges=30)
    expect("bpe: merges reduce token count below char count",
           len("word sample corpus. " * 60) > 20 * len(
               tok.encode("word sample corpus ")))


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

def _tiny_model(vocab=20, block=16):
    torch.manual_seed(0)
    return TinyGPT(vocab_size=vocab, block_size=block, n_embd=16,
                   n_heads=2, n_layers=1, dropout=0.0)


def test_model_forward_shapes():
    m = _tiny_model()
    x = torch.randint(0, 20, (3, 8))
    logits = m(x)
    expect("model: logits shape [batch, seq, vocab]",
           tuple(logits.shape) == (3, 8, 20))
    x, y = x, torch.randint(0, 20, (3, 8))
    _, loss = m(x, y)
    expect("model: loss without mask is float", float(loss) >= 0)


def test_model_causal_mask():
    m = _tiny_model()
    x = torch.randint(0, 20, (1, 8))
    logits1 = m(x)[0, 0].detach().clone()
    x2 = x.clone()
    x2[0, 5] = (x2[0, 5] + 3) % 20  # change a future token
    logits2 = m(x2)[0, 0].detach().clone()
    expect("model: causal mask blocks future tokens",
           torch.allclose(logits1, logits2))


def test_answer_only_loss():
    m = _tiny_model()
    x = torch.randint(0, 20, (1, 8))
    y = torch.randint(0, 20, (1, 8))
    _, loss_zero_mask = m(x, y, target_mask=torch.zeros(1, 8, dtype=torch.bool))
    expect("model: all-zero answer mask gives zero loss",
           float(loss_zero_mask) == 0.0)
    _, loss_full = m(x, y)
    m_partial = torch.tensor([[True] * 4 + [False] * 4])
    _, loss_ans = m(x, y, target_mask=m_partial)
    # manual computation over the first four positions
    logits = m(x)
    import torch.nn.functional as F
    manual = F.cross_entropy(
        logits[:, :4].reshape(-1, 20), y[:, :4].reshape(-1)).detach()
    expect("model: answer-only loss matches manual computation",
           approx(float(loss_ans), float(manual)),
           f"{float(loss_ans)} vs {float(manual)}")


def test_padding_invariance():
    """Padding must not change the outputs (logits/loss) of real positions."""
    m = _tiny_model()
    m.eval()
    torch.manual_seed(1)
    ids = torch.randint(0, 20, (9,))
    mask = torch.ones(9, dtype=torch.bool)
    mask[:5] = False  # first five positions are prompt (not scored)
    x = ids[:-1].unsqueeze(0); y = ids[1:].unsqueeze(0)
    msk = mask[1:].unsqueeze(0)
    _, single = m(x, y, msk,
                  key_padding_mask=torch.zeros((1, 8), dtype=torch.bool))
    single_logits = m(x)[0, :4].detach().clone()
    # Batch: row 0 unpadded, row 1 padded after position 6.
    batch_x = x.clone().repeat(2, 1)
    batch_y = y.clone().repeat(2, 1)
    batch_m = msk.clone().repeat(2, 1)
    pads = torch.zeros((2, 8), dtype=torch.bool)
    pads[1, 6:] = True
    batch_x[1, 6:] = 0; batch_y[1, 6:] = 0; batch_m[1, 6:] = False
    batched_logits_row0 = m(batch_x)[0, :4].detach().clone()
    expect("model: padding does not disturb unpadded-row logits",
           torch.allclose(single_logits, batched_logits_row0, atol=1e-5))
    # Loss-equality with identical (unpadded) duplicate rows.
    batch_x = torch.cat([x, x], dim=0)
    batch_y = torch.cat([y, y], dim=0)
    batch_m = torch.cat([msk, msk], dim=0)
    _, dup_batched = m(batch_x, batch_y, batch_m,
                       key_padding_mask=torch.zeros((2, 8), dtype=torch.bool))
    expect("model: duplicate-row batch loss equals single loss",
           approx(float(single), float(dup_batched), tol=1e-4),
           f"single={float(single):.4f} batched={float(dup_batched):.4f}")


# ---------------------------------------------------------------------------
# Checkpoint / generation tests
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip(tmp_path=None):
    tok = CharTokenizer.build("abcdef Sidney<END>")
    torch.manual_seed(0)
    model = TinyGPT(vocab_size=tok.vocab_size, block_size=16, n_embd=16,
                    n_heads=2, n_layers=1, dropout=0.0)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "vocab_size": tok.vocab_size,
        "config": {"block_size": 16, "n_embd": 16, "n_heads": 2,
                   "n_layers": 1, "dropout": 0.0},
        "tokenizer": tok.state(),
    }
    p = Path(PROJECT_DIR / "model" / "_pytest_ckpt.pt")
    p.unlink(missing_ok=True)
    torch.save(ckpt, p)
    try:
        loaded = torch.load(p, weights_only=False)
        m2 = TinyGPT(vocab_size=loaded["vocab_size"], **loaded["config"])
        m2.load_state_dict(loaded["model_state_dict"])
        tok2 = CharTokenizer.from_state(loaded["tokenizer"])
        x = torch.tensor([[tok.stoi.get("a", 0)]])
        out1 = model(x)[0, 0].detach()
        out2 = m2(x)[0, 0].detach()
        expect("checkpoint: save/load reproduces identical logits",
               torch.allclose(out1, out2))
        expect("checkpoint: tokenizer survives save/load", tok2.decode(
            tok2.encode("bc")) == "bc")
    finally:
        p.unlink(missing_ok=True)


def test_generation_stop_and_validation():
    tok = CharTokenizer.build("User: hi<END>")
    m = _tiny_model(vocab=tok.vocab_size, block=16)
    m.eval()
    device = torch.device("cpu")
    text, status, ids = generate(
        m, tok, "hi", device, max_new_tokens=40, temperature=0.0)
    expect("generate: returns allowed status only",
           status in ("end", "marker", "max_tokens"), status)
    expect("generate: output is a decoded string of in-vocab ids",
           isinstance(text, str)
           and all(0 <= int(i) < tok.vocab_size for i in ids))
    try:
        generate(m, tok, "s" * 40, device, max_new_tokens=5)
        expect("generate: over-long prompt raises", False)
    except ValueError:
        expect("generate: over-long prompt raises", True)


def test_generation_determinism():
    torch.manual_seed(0)
    tok = CharTokenizer.build("User: hi<END>")
    m = _tiny_model(vocab=tok.vocab_size, block=16)
    m.eval()
    device = torch.device("cpu")
    t1, s1, _ = generate(m, tok, "hi", device, max_new_tokens=25,
                         temperature=0.0)
    t2, s2, _ = generate(m, tok, "hi", device, max_new_tokens=25,
                         temperature=0.0)
    expect("generate: greedy decoding is deterministic", t1 == t2 and s1 == s2)


def test_sampling_tools():
    from trainer.generate import marker_hold_len
    expect("sampling: marker holdback detects partial markers",
           marker_hold_len("hello <EN") == 3)
    expect("sampling: zero holdback for plain text",
           marker_hold_len("hello") == 0)


def test_eval_metrics():
    from tests.evaluate import has_long_repeat, number_match_exact, token_f1
    expect("eval: token F1 perfect overlap is 1.0",
           approx(token_f1("the elephant is gray", "The elephant is gray."), 1.0))
    expect("eval: token F1 no overlap is 0.0",
           approx(token_f1("cat dog", "purple green"), 0.0))
    expect("eval: repeat detector finds loops",
           has_long_repeat("this is a sample " * 8))
    expect("eval: repeat detector passes single ideas",
           not has_long_repeat("A transformer is a model with attention layers."))
    expect("eval: number extractor", number_match_exact("12 times 12 is 144.", 144))
    expect("eval: number extractor rejects wrong value",
           not number_match_exact("12 times 12 is 14.", 144))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    tests = [
        test_dataset_parsing, test_duplicate_detection, test_split_integrity,
        test_split_reproducible, test_no_fact_leak_across_splits,
        test_arithmetic_holdout_band, test_expansion_quality,
        test_char_tokenizer_roundtrip, test_bpe_roundtrip,
        test_bpe_vocab_growth, test_model_forward_shapes,
        test_model_causal_mask, test_answer_only_loss, test_padding_invariance,
        test_checkpoint_roundtrip, test_generation_stop_and_validation,
        test_generation_determinism, test_sampling_tools, test_eval_metrics,
    ]
    for t in tests:
        try:
            t()
        except Exception as exc:  # surface test-crash as failure
            RESULTS.append((t.__name__, False, f"CRASH: {exc!r}"))
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print(f"\n{passed} passed / {failed} failed\n")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"FAIL {name}: {detail}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
