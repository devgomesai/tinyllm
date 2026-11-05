# Model versions (manifest)

Naming: `model/v1/` = preserved baselines from the original repository.
`model/v2/<tokenizer>/` = TinyGPT 2.0 experiments trained on the cleaned,
example-split dataset. Files inside each run directory:

- `best.pt` — lowest full-split validation loss during the run (use this)
- `final.pt` — weights at the last step
- `training_log.csv` — step / train loss / val loss / lr / time
- `train_summary.json` — run metadata incl. peak GPU memory

Baselines are never overwritten by training; each new run goes to its own
directory (override with `--out-dir`).

| Version | Path | Dataset | Recipe | Params | Val answer-loss | Notes |
| --- | --- | --- | --- | ---: | ---: | --- |
| v1 | `v1/mini_gpt_v1.pt` | data_v1.txt | char, v1 script | 4,815,958* | n/a | original exact checkpoint, preserved |
| v1 | `v1/mini_gpt_v2.pt` | data.txt | char, v1 script (90% char split) | 4,815,958 | 0.4821 (subset) | baseline for all v2 comparisons |
| v2 | `v2/char/best.pt` | processed splits + expansion | char, example-padded, 4L/256d/8h, ctx 256, dropout 0.1 | TBD | TBD | TinyGPT 2.0 |
| v2 | `v2/bpe/best.pt` | processed splits + expansion | BPE vocab ~400 merges, example-padded | TBD | TBD | TinyGPT 2.0 experimental |

\* The original README claimed 633,558 parameters; the actual default
constructor of the trained checkpoint (256 dim / 8 heads / 6 layers) is
4,815,958. The evaluation reports measured values only.

Checkpoint format (v2): `model_state_dict`, `config`, `vocab_size`,
`tokenizer` (self-describing), legacy `stoi`/`itos` for char models, and
`meta` (steps, val loss, split sizes, seed, timestamps). The CLI and the
evaluation auto-detect the tokenizer type, so v1 checkpoints keep working.
