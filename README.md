# TinyLLM (TinyGPT 2.0)

A small GPT-style language model built from scratch in PyTorch, trained on a
corpus of question-and-answer dialogues, with a char-level model, an
experimental BPE model, and a reproducible evaluation framework.

TinyLLM is a learning and experimentation project. It imitates the shape of a
conversation in its training data, but it is not a production LLM and not a
real assistant: it predicts the next character (or next subword token) of a
text, nothing more. Do not rely on anything it says.

## Table of Contents

- [Quick start: full workflow](#quick-start-full-workflow)
- [Train and test cheat sheet](#train-and-test-cheat-sheet)
- [What is new in 2.0](#what-is-new-in-20)
- [Char vs BPE models](#char-vs-bpe-models)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Text Generation CLI](#text-generation-cli)
- [Results](#results)
- [Checkpoint](#checkpoint)
- [Project Structure](#project-structure)
- [GPU requirements](#gpu-requirements)
- [Known limitations](#known-limitations)
- [Technologies](#technologies)

## Quick start: full workflow

Run these from the project root, in order. Each stage depends on the previous
one; commands are idempotent and seeded, so rerunning them reproduces the same
artifacts.

```powershell
# 0. One-time environment setup (Python 3.13 + PyTorch CUDA build)
uv sync

# 1. Build the dataset: parse -> expand -> clean -> stratified splits
uv run python -m trainer.dataset --seed 42

# 2. Run the automated test suite (48 checks; exits non-zero on failure)
uv run python tests/test_pipeline.py

# 3. Train the recommended model (char tokenizer, 8k steps)
uv run python trainer/train.py --tokenizer char --iters 8000
#    -> model/v2/char/best.pt   (also: training_log.csv, train_summary.json)

# 4a. Score it against the held-out suites and compare with the preserved baseline
uv run python tests/evaluate.py --model model/v2/char/best.pt
# 4b. (score the preserved baseline the same way, for comparison)
uv run python tests/evaluate.py --model model/v1/mini_gpt_v2.pt

# 5a. Chat with it interactively
uv run python tests/testllm.py --model model/v2/char/best.pt
# 5b. or one prompt and exit
uv run python tests/testllm.py --model model/v2/char/best.pt --prompt "Explain how a neural network learns" --temperature 0.7 --tokens 200
# 5c. fully deterministic (greedy) generation
uv run python tests/testllm.py --model model/v2/char/best.pt --greedy --prompt "What is 7 + 8?"
```

Optional variants:

```powershell
# Train the experimental BPE model (larger batch because BPE tokens are ~4x longer)
uv run python trainer/train.py --tokenizer bpe --iters 8000 --batch-size 96
uv run python tests/evaluate.py --model model/v2/bpe/best.pt
uv run python tests/testllm.py --model model/v2/bpe/best.pt

# Retrain the baseline recipe without touching the original checkpoints
# (writes a copy you can compare against; use --out-dir to keep runs apart)
uv run python trainer/train.py --tokenizer char --iters 8000 --n-layers 6 --block-size 128 --out-dir model/v2/char_baseline_recipe
```

Small runs to check the plumbing before a real training (2 minutes, ~0.5 GB):

```powershell
uv run python trainer/train.py --smoke --tokenizer char   # 30-iter smoke run
uv run python trainer/train.py --smoke --tokenizer bpe
```

## Train and test cheat sheet

Each run gets its own `--out-dir` so nothing is overwritten.

```powershell
# Train: char model (recommended)
uv run python trainer/train.py --tokenizer char --iters 8000 --out-dir model/v2/char_run3

# Train: BPE model
uv run python trainer/train.py --tokenizer bpe --iters 8000 --batch-size 96 --out-dir model/v2/bpe_run2

# Quick plumbing check (30 iters)
uv run python trainer/train.py --smoke --tokenizer char --out-dir model/smoke_char

# Score a checkpoint on the held-out suites (writes eval_results/*.json)
uv run python tests/evaluate.py --model model/v2/char_run3/best.pt

# Chat (interactive, no --prompt)
uv run python tests/testllm.py --model model/v2/char_run3/best.pt

# One answer, sampled with a fixed temperature
uv run python tests/testllm.py --model model/v2/char_run3/best.pt --prompt "What is a neuron?" --temperature 0.7 --tokens 200

# One answer, deterministic
uv run python tests/testllm.py --model model/v2/char_run3/best.pt --greedy --prompt "What is a neuron?"
```

To compare runs, check `best_val_loss` in each `train_summary.json` and the
metrics in the matching `eval_results/` JSON. Use the same `--seed` and dataset.

## Which stage writes what

| Stage | Command | Output |
| --- | --- | --- |
| dataset | `python -m trainer.dataset` | `data/processed/{train,val,test}.txt`, `stats.json`, `pipeline_report.json` |
| tests | `python tests/test_pipeline.py` | console report (48 assertions) |
| training | `trainer/train.py` | `model/v2/<tokenizer>/best.pt`, `final.pt`, `training_log.csv`, `train_summary.json` |
| evaluation | `tests/evaluate.py` | `eval_results/eval_<model>_<tokenizer>_<timestamp>.json` |
| chat | `tests/testllm.py` | terminal replies (no artifacts) |

## What is new in 2.0

Compared to the v1 repository (character-level model, 90/10 character-position
split, random wholesale windows, single metric):

- **Example-level dataset pipeline** (`trainer/dataset.py`): parsing, cleaning,
  fact-grouping and deterministic example-level splits. Paraphrased variants of
  one fact never cross splits any more.
- **Arithmetic hold-out bands**: operand combinations where both operands are
  >= 10 are reserved for the test set.
- **Expanded, categorized corpus** (`trainer/data_expansion.py`): Python,
  PyTorch, neural nets, CNNs, Transformers, tokenization, ML, English grammar,
  debugging, program output prediction, follow-up dialogues and programmatic
  arithmetic.
- **Refactored trainer** (`trainer/train.py`): example-padded batches with
  attention masks, configurable architecture, cosine schedule with warmup,
  best+final checkpoints, CSV logs, graceful CUDA OOM handling.
- **BPE tokenizer from scratch** (`trainer/tokenizer.py`), trained on the
  cleaned corpus, with `<END>` / `<PAD>` special tokens.
- **Better generation** (`trainer/generate.py` + improved
  `tests/testllm.py`): temperature, top-k, top-p, repetition penalty, greedy
  mode, context validation, deterministic stopping.
- **Evaluation framework** (`tests/evaluate.py`): five held-out categories
  (memorization, rewording, unseen topics, reasoning, arithmetic) plus
  train/val/test losses, with JSON reports.

## Char vs BPE models

Both models use the same transformer. They differ only in how text is split
into tokens, the units the model predicts.

| | Char model (`--tokenizer char`) | BPE model (`--tokenizer bpe`) |
| --- | --- | --- |
| Token | one character (`T`, `o`, ` `, `?`) | a learned chunk, e.g. `neural`, `ing`, ` the` |
| Vocabulary | 86 characters from the corpus | 86 characters + 400 learned merges + 2 specials = 488 |
| Tokenizer training | none, the mapping is fixed | greedy merges of the most frequent adjacent pair, from scratch in `trainer/tokenizer.py` |
| Sequence length | long: each character is a token | short: chunks cover more text per token |
| Strength | simple; any text in the alphabet is representable | a 256-token window holds more text; learns common word pieces |
| Weakness | the model must learn spelling letter by letter | each chunk is seen less often; more chances to overfit small data |
| Batch default | 32 | 96 (shorter sequences use less memory per example) |
| Best val loss here | 1.4700 (step 1250) | 2.6491 (step 500) |

Read the loss columns carefully. Char and BPE losses are per token, and their
tokens are different units, so the numbers cannot be ranked against each
other. Compare models with `tests/evaluate.py`, which scores the same held-out
questions for both.

The shipped recommendation is the char model: it is the one with measured
held-out results. The BPE model is kept as an experiment.

## Architecture

```
TinyGPT (trainer/model.py)
├── token embedding       (vocab_size × d_model)
├── positional embedding  (block_size × d_model)
├── nn.TransformerEncoder (n_layers × [causal MHSA + GELU FFN, norm_first])
├── final LayerNorm
└── lm head               (d_model × vocab_size)
```

- Causal self-attention: position *t* sees only positions <= *t*.
- The v1 model and the v2 char model share exactly this architecture and
  parameter names, so v1 checkpoints still load and generate.
- The BPE model uses the same trunk with a token vocabulary instead of a
  character vocabulary, and supports padded batches via
  `src_key_padding_mask`.

### Model sizes

| Model | Vocab | d_model | heads | layers | block | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (`model/v1/mini_gpt_v2.pt`, v1 recipe) | 86 chars | 256 | 8 | 6 | 128 | 4,815,958 |
| char (new pipeline, char) | 86 chars | 256 | 8 | 4 | 256 | 3,269,206 |
| bpe (new pipeline, BPE, 400 merges) | 488 | 256 | 8 | 4 | 256 | 3,475,432 |

Dataset prompt + answer always fits the 256-token context of the v2 models
(the longest formatted example in the final corpus is below that limit), so
nothing is truncated.

## Dataset

The raw corpora live in `data/`:

- `data.txt` — the v2 dialogue corpus (3,432 `User: / TinyLLM: / <END>`
  examples, 398,922 chars before expansion). Raw, never modified by tooling.
- `data_v1.txt` — the v1 corpus, kept for reference.
- `processed/train.txt`, `processed/val.txt`, `processed/test.txt` — generated
  by the pipeline below; do not edit by hand.

Format of every example:

```text
User: What is a neuron?
TinyLLM: It is a small unit that takes numbers in, combines them, and passes a number on.
<END>
```

### Dataset audit findings (v1.4e data)

- 3,432 examples, all structurally well-formed; zero exact duplicates and zero
  repeated normalized questions at the surface level.
- But only ~1,447 distinct underlying facts: 664 facts carry 3-6 question
  wrappers ("Can I ask, what is X?", "I was wondering, what is X?", "Please
  tell me, what is X?"...). This wrapper duplication is why trained models
  answer questions that look like the corpus and fail on other wording.
- The v1 90/10 split fell at a character position, so paraphrases of one fact
  routinely straddled the train/val boundary (validation loss leakage).
- ~775 examples exceeded the v1 128-character context in formatted form.

## Data preparation

```powershell
uv run python -m trainer.dataset --seed 42
```

This does, deterministically:

1. Parse `data/data.txt` into examples (malformed blocks are counted, skipped).
2. Generate the expansion corpus (`trainer/data_expansion.py`), verified to
   contain no question that already exists in `data.txt` and to stay inside the
   v1 character vocabulary.
3. Dedupe exact pairs and identical normalized questions.
4. Group examples by base fact (wrapper prefixes stripped), so the same fact is
   never split across sets, then stratify the assignment by category into
   80% train / 10% val / 10% test with the given seed.
5. Force both-operands->=10 arithmetic (`__arith_large__`) into the test set.
6. Write `data/processed/{train,val,test}.txt` + `stats.json` +
   `pipeline_report.json`.

Split sizes (seed 42, final corpus): 3,560 train (1,664 fact groups) /
449 val (207 groups) / 462 test (219 groups), 4,471 examples total —
see `data/processed/stats.json` for the per-category breakdown.

## Training

```powershell
# character-level model (recommended first)
uv run python trainer/train.py --tokenizer char --iters 8000

# BPE (experimental; larger batch because BPE sequences are ~4x shorter in tokens)
uv run python trainer/train.py --tokenizer bpe --iters 8000 --batch-size 96
```

Validation loss bottoms long before the run end (~step 1250 in practice, see
Results), so iterations afterwards are just `best.pt` insurance. Each run
writes to `model/v2/<tokenizer>/` by default — pass a distinct `--out-dir`
when re-running, otherwise the existing run's files are overwritten.

Both write to `model/v2/char/` or `model/v2/bpe/` and save `best.pt`
(lowest validation loss), `final.pt`, `train_summary.json` and
`training_log.csv`. The v1 baseline (`model/v1/mini_gpt_v2.pt`) is never
overwritten.

| Option | Default | Description |
| --- | ---: | --- |
| `--tokenizer` | `char` | `char` or `bpe` |
| `--iters` | 8000 | optimizer steps |
| `--batch-size` | 32 | examples per step (auto-halved on CUDA OOM) |
| `--block-size` | 256 | context length in tokens |
| `--lr` | 6e-4 | peak AdamW learning rate |
| `--warmup-iters` | 200 | linear LR warmup steps |
| `--n-embd` | 256 | model width |
| `--n-heads` | 8 | attention heads |
| `--n-layers` | 4 | transformer layers |
| `--dropout` | 0.1 | regularization |
| `--eval-interval` | 250 | steps between evals |
| `--seed` | 42 | all randomness |
| `--out-dir` | `model/v2/<tokenizer>` | checkpoint directory |

Notes:

- Validation loss is computed over the **entire** validation split
  (deterministic), not on random batches.
- Examples longer than `--block-size` are excluded with a warning, never
  silently truncated.
- Test loss is only computed by `tests/evaluate.py`, after training decisions
  are done.

## Evaluation

```powershell
# default: evaluate the baseline checkpoint
uv run python tests/evaluate.py

# evaluate one of the new models
uv run python tests/evaluate.py --model model/v2/char/best.pt
uv run python tests/evaluate.py --model model/v2/bpe/best.pt
```

Writes a JSON report to `eval_results/` with:

- answer-masked loss on train / val / test (test used for reporting only)
- A memorization: training questions, greedy decode
- B rewording: held-out questions re-worded in forms absent from all splits
- C unseen topics: probes verified absent from the whole corpus
- D reasoning: unseen sequences (step 13-25; training had 1-9), Python
  program-output prediction, number comparison
- E arithmetic: unseen two-operand combinations in three difficulty tiers, plus
  the reserved large-band examples from the original corpus

Per-suite metrics: completion rate, `<END>` rate, repetition rate, malformed
rate, token-F1 vs reference (A, B), exact match (D, E).

## Text Generation CLI

```powershell
uv run python tests/testllm.py                                  # interactive, baseline
uv run python tests/testllm.py --model model/v2/char/best.pt    # new char model
uv run python tests/testllm.py --prompt "Explain how a neural network learns" --temperature 0.7 --tokens 200
uv run python tests/testllm.py --prompt "What is 7 + 8?" --temperature 0
```

| Option | Default | Description |
| --- | ---: | --- |
| `--model` | `model/v1/mini_gpt_v2.pt` | any checkpoint (char or BPE, auto-detected) |
| `--prompt` | none | generate once and exit |
| `--tokens` | 200 | max new tokens/characters |
| `--temperature` | 0.6 | 0 = greedy |
| `--top-k` | 10 | 0 disables |
| `--top-p` | 0 | nucleus sampling in (0,1); 0 disables |
| `--repetition-penalty` | 1.0 | >1 discourages already-used tokens |
| `--greedy` | off | deterministic decoding |

Unknown prompts characters that are not in the vocabulary are skipped with a
counter (char model); BPE handles them through its base alphabet. Prompts
longer than the model context are rejected with a clear error. Generation stops
at `<END>`; a `\nUser:` / `\nTinyLLM:` header inside an answer also terminates
it, so the reply never invents a new conversation turn.

## Results

Full JSON reports live in `eval_results/` and `model/v2/*/train_summary.json`.
All generation metrics use
greedy decoding (deterministic) on the same fixed suites (seed 42).

Baseline (`model/v1/mini_gpt_v2.pt`, v1 recipe on `data.txt`, greedy):

| Metric | Value |
| --- | ---: |
| Train answer-loss (examples fitting ctx 128) | 0.3022 |
| Val answer-loss | 0.4821 |
| Held-out test answer-loss | 0.3505 |
| A memorization token-F1 | 0.791 |
| B reworded token-F1 | 0.856* |
| Unseen sequences exact-match | 0.000 |
| Program-output prediction (token-F1) | 0.000 |
| Unseen arithmetic (both operands >= 10) exact-match | 0.026 |
| **Seen** large-band arithmetic exact-match | **0.971** |
| Unseen-topic relevant-word rate | 0.000 |

The pattern is pure memorization: 97% on arithmetic operand combinations it
saw during training, ~0% on held-out combinations, fluent-but-off-topic text
for unseen topics ("What is a sundial?" -> "Australia is the smallest
continent."). Caveat: the baseline was trained on ~90% of the raw corpus by
character position, so its B suite (and part of the test split) was already
seen; the v2 models are trained only on the clean train split, making the B
comparison conservative in the baseline's favor.

char v2 (`model/v2/char/best.pt`, best val at step 1250/8000, greedy):

| Metric | Value |
| --- | ---: |
| Val answer-loss (all 449 val examples fit ctx 256) | 1.3444 |
| Held-out test answer-loss | 1.2249 |
| A memorization token-F1 | 0.232 |
| B reworded token-F1 | 0.123 |
| Unseen arithmetic (both operands >= 10) exact-match | 0.051 |
| Reserved data.txt large arithmetic exact-match | 0.200 |
| Unseen sequences exact-match | 0.000 |
| Unseen number comparison exact-match | 0.000 |
| Program-output token-F1 | 0.080 |
| Unseen-topic relevant-word rate | 0.000 |
| Clean terminations (ends with `<END>`, no truncation/malformed) | 100% |

The v2 char loss numbers are not a regression against the baseline: the
baseline's 0.4821 val loss was measured on data it had effectively seen, and
the v2 model has never seen its val set. What the new run buys is *honest*
held-out measurement — and it shows the model is still far from generalizing
(answers are fluent-looking but factually wrong outside training content, e.g.
"What is a capacitor?" -> "A copatoin is a partor stores keys..."). 100% of
generations terminate cleanly within budget, which the old pipeline did not
guarantee. Longer training does not help here: validation loss bottomed at
step 1250 and rose monotonically afterwards (classic small-corpus overfitting),
which is why `best.pt` — not `final.pt` — is the shipped checkpoint. A
repeat run with the same 8k recipe reproduced the pattern: best val 1.4700,
so no further steps were warranted.

BPE v2 (`model/v2/bpe/best.pt`, 8000 iters, 400 merges, 488-token vocab):

| Metric | Value |
| --- | ---: |
| Best val answer-loss | 2.6491 (step 500) |
| Peak GPU memory / OOM events | 0.98 GB / 0 |

BPE's val loss is not comparable to char's: the token vocabulary differs, so
per-token loss is measured over different units. Its best step is 500, far
earlier than char's 1250, and it rose after that. Generation metrics
(`tests/evaluate.py`) have not been run on it yet.

## Checkpoint

Checkpoint contents: `model_state_dict`, `config`, `vocab_size`,
`tokenizer` (self-describing: type, vocab, merges or char mappings; legacy
`stoi`/`itos` keys included), and `meta` (step, val loss, split sizes).

`tests/testllm.py` auto-detects the tokenizer type, so v1 checkpoints keep
working with the new CLI.

## Project Structure

```text
TinyLLM/
├── data/
│   ├── data.txt               # raw dialogue corpus (source of truth)
│   ├── data_v1.txt            # v1 corpus, reference only
│   └── processed/             # generated splits + stats (do not edit)
├── model/
│   ├── v1/                    # preserved baselines (never overwritten)
│   │   ├── mini_gpt_v1.pt     # v1 exact checkpoint (data_v1.txt)
│   │   └── mini_gpt_v2.pt     # v1-recipe checkpoint (data.txt)
│   └── v2/                    # TinyGPT 2.0 experiments
│       ├── char/              # best.pt, train_summary.json, training_log.csv
│       └── bpe/               # best.pt, final.pt, train_summary.json, training_log.csv
├── trainer/
│   ├── dataset.py             # parse, clean, group, split, write
│   ├── data_expansion.py      # authored + programmatic expansion corpus
│   ├── tokenizer.py           # char tokenizer + from-scratch BPE
│   ├── model.py               # TinyGPT architecture
│   ├── generate.py            # sampling + generation logic
│   └── train.py               # refactored training loop
├── tests/
│   ├── testllm.py             # generation CLI
│   ├── evaluate.py            # evaluation framework
│   └── test_pipeline.py       # automated tests (uv run python tests/test_pipeline.py)
├── scripts/                   # one-off dataset audit scripts (phase 1)
├── eval_results/              # evaluation JSON reports
└── README.md
```

## GPU requirements

Developed and tested on a single NVIDIA RTX 3060 12 GB (PyTorch CUDA 12.8
build). The v2 models train comfortably in under 1 GB of GPU memory at the
default settings; peak memory is logged in `train_summary.json`. CPU works but
is slow.

## Known limitations

- This is a small model trained from scratch on a few thousand short
  dialogues; it has no world model and no reasoning ability beyond fragile
  pattern completion. It will confidently produce wrong facts.
- Arithmetic generalization beyond memorized operand bands is genuinely hard
  for this architecture at this scale; the evaluation measures how much it
  transfers, not whether it is reliable.
- The corpus stays small; expanded categories are thin compared to a real
  instruction-tuning dataset.
- Generation quality depends heavily on sampling settings; greedy decoding is
  the only reproducible mode.
- Identity facts in the dataset ("Who made you?") describe this project; the
  model repeats them as text, nothing more.

## Technologies

- Python 3.13, PyTorch (CUDA 12.8 build), uv for dependency management,
  Rich for the CLI. No other runtime dependencies: the BPE tokenizer and BPE
  trainer are part of `trainer/tokenizer.py`.
