# TinyLLM (TinyGPT 2.0)

A small GPT-style language model built from scratch in PyTorch, trained on a
corpus of question-and-answer dialogues, with a char-level model, an
experimental BPE model, and a reproducible evaluation framework.

TinyLLM is a learning and experimentation project. It imitates the shape of a
conversation in its training data, but it is not a production LLM and not a
real assistant: it predicts the next character (or next subword token) of a
text, nothing more. Do not rely on anything it says.

## Table of Contents

- [What is new in 2.0](#what-is-new-in-20)
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
| char (new pipeline, char) | TBD | 256 | 8 | 4 | 256 | TBD |
| bpe (new pipeline, BPE) | TBD | 256 | 8 | 4 | 256 | TBD |

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

Split sizes (seed 42, final corpus): TBD — see `data/processed/stats.json`.

## Training

```powershell
# character-level model (recommended first)
uv run python trainer/train.py --tokenizer char --iters 8000

# BPE (experimental)
uv run python trainer/train.py --tokenizer bpe --bpe-merges 400 --iters 8000
```

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
| `--out-dir` | `model/exp_<tokenizer>` | checkpoint directory |

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

See `eval_results/*.json` for full numbers. Summary table: TBD after final runs.

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
│   ├── VERSIONS.md            # manifest of every checkpoint
│   ├── v1/                    # preserved baselines (never overwritten)
│   │   ├── mini_gpt_v1.pt     # v1 exact checkpoint (data_v1.txt)
│   │   └── mini_gpt_v2.pt     # v1-recipe checkpoint (data.txt)
│   └── v2/                    # TinyGPT 2.0 experiments
│       ├── char/              # best.pt, final.pt, logs
│       └── bpe/               # best.pt, final.pt, logs
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
