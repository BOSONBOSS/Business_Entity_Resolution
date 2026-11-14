# Amazon ML Challenge 2026 — Business Entity Resolution

**Read [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) before writing or modifying any code.**
[HACKATHON_RULES.md](HACKATHON_RULES.md) holds the organiser logistics. The rules below are
binding on every piece of code, notebook, and document in this repo.

## The task in one paragraph

Given business records from 3 noisy sources with no shared identifiers, find, **for every Source 1
entity**, all matching records from Source 2 and Source 3. Source 1 is the deduplicated reference.
An entity may match zero, one, or many records. Scored by **macro-averaged F_0.5** (precision-heavy)
per Source 1 entity, singletons included.

## Fatal mistakes — any one of these costs everything

1. **Reading or writing a TSV without `sep="\t"`.** It fails *silently* into one column. Every
   read/write passes the separator explicitly. No exceptions.
2. **Hard-coding country to `{US, India}`.** The test set contains **France**, which never appears
   in training. Country is an **open set of strings**. Every test entity, France included, must be
   in the submission. Any country-conditioned branch needs a working default.
3. **Dropping or duplicating Source 1 rows.** Exactly one row per test Source 1 entity — missing
   or duplicate rows cause **rejection**, not a low score.
4. **Emitting an ID that isn't in the test set, or an `S1-` self-match.** Rejection.
5. **Duplicate IDs inside one ID list.** Rejection.
6. **Any external lookup** — geocoding APIs, business registries, web scraping, internet data
   augmentation. **Instant disqualification**, and code is reviewed. The pipeline must run with
   networking disabled.
7. **A final model that is not MIT/Apache-2.0 licensed or exceeds 8B parameters.** Check the licence
   before downloading weights.
8. **Uploading without running the validator.** `utils/validate_submission.py` is free; a rejected
   upload burns one of only 15 submission slots.

## ⚠️ Never subsample negatives when EVALUATING

Subsampling negatives is a legitimate class-imbalance treatment **for training**. Using it for
validation or test scoring inflates the score badly, because every candidate not scored is a
false positive that cannot happen.

Measured on this project: scoring ~43 of each entity's ~200 candidates reported **0.9639**;
scoring all 200 reported **0.9271** — a **0.0368** overstatement, larger than every genuine
improvement combined. Recall was identical (positives are never subsampled); the whole loss was
precision, which F_0.5 weights double.

It hits singletons hardest (0.9514 → 0.7881): an entity with no true matches has *only* negatives
among its candidates, so multiplying negatives multiplies its chances of a wrong prediction.

**Rules:**
- `--neg-per-entity N` for `--fold train`. **`--neg-per-entity 0` for val and test, always.**
- A threshold tuned on a subsampled set is tuned for the wrong precision/recall balance and will
  be too low. Re-tune on the full candidate set.
- Any evaluation must score exactly what inference scores. If the two differ, the number is
  fiction.

## Metric-driven design rules

- **F_0.5 weights precision 2× over recall.** When the model is uncertain, **predict nothing**.
  A false merge costs more than a missed match.
- **Macro-average over entities:** every Source 1 entity counts equally, whether it has 0 or 10
  matches. A mistake on a singleton costs as much as a mistake on a large cluster.
- **Singletons score 1.0 when correctly predicted empty, 0.0 if you predict anything.** They are
  free points. Never force a match just because candidates exist.
- **Tune the decision threshold against macro-F_0.5 on held-out data** — not pair-level accuracy,
  not AUC, not the public leaderboard.
- **Blocking recall is the hard ceiling on final recall.** Measure candidate-stage recall separately
  and treat it as a first-class metric.

## Two output files, always produced together

- `output/matching_results.tsv` — final matches; the only leaderboard-scored file.
- `output/candidate_pairs.tsv` — the candidate set **actually fed to the matching model at
  inference** (the last blocking stage, not an earlier pass). Final matches must be a **subset**.

## Coding standards this repo enforces

- **Docstrings on every function** — the code is a reviewed submission artefact, not just ours.
  One-line summary + Args + Returns, minimum.
- **Seed everything** via `src/utils.set_seed()`; record the seed in the run config.
- **No hard-coded paths or magic numbers in logic.** Paths and hyperparameters live in
  `configs/*.yaml`, snapshotted per run via `src/utils.snapshot_run()`.
- **Blocking, feature building, and scoring are separate, importable stages**, each runnable on
  train and test with identical code. Duplicated train/inference logic is how leaks and format
  bugs happen.
- **Fixed validation split, chosen once, never changed mid-competition.**
- **Log every experiment** to `experiments/log.md` the moment it finishes: id, config diff, local
  macro-F_0.5, blocking recall, LB score if submitted, one-line takeaway. This file is the raw
  material for the required methodology document.
- **Archive every uploaded file** to `submissions/` — never overwrite. Version history is required.

## Machine constraints (this laptop)

Lenovo LOQ · i9-14900HX, 24 cores / 32 threads · 32 GB RAM · RTX 4060 Laptop, 8 GB VRAM ·
torch 2.6.0+cu124, pandas 2.3.3, Python 3.10.

- **The dGPU drops off the bus on battery power.** On battery, `nvidia-smi` fails,
  `Present: False`, and `torch.cuda.is_available()` returns `False`. **Stay on AC power for any
  CUDA work** — unplugging mid-run kills the job. Verify with
  `python -c "import torch;print(torch.cuda.is_available())"` before starting a long run.
- **RAM is the binding constraint, not disk or CPU.** Baseline usage sits near the commit limit
  with two browsers open. Close them before a big run. Never load all three test files into
  object-dtype pandas at once (~6–8 GB); use Polars or pyarrow-backed strings, and stream or chunk.
- **8 GB VRAM caps model choice** well below the competition's 8B-parameter ceiling. An 8B model
  needs ~16 GB at fp16 and is not runnable here — and encoding ~12M records with one is not
  feasible in a 3-day window on any laptop. Think 100M–600M parameter encoders at most.
- **Prefer CPU-bound approaches.** 32 threads on `rapidfuzz` + LightGBM is the strong suit here;
  treat GPU embeddings as an accelerator for one stage, not the backbone.

## Before writing code, confirm

- Does this touch the submission? Is the 5-per-day slot budget accounted for?
- Does it reference the network, an external dataset, or a non-permissive model?
- Does it survive an unseen `country` value?
- Is it reproducible from a clean checkout (seed + config + dataset path)?

## Layout

```
dataset/train/  dataset/test/   official data, paths mirrored verbatim from the spec
work/                           derived intermediates (folds, blocks, features) — gitignored
output/                         matching_results.tsv + candidate_pairs.tsv
utils/                          validate_submission.py (from the student_resource bundle)
src/                            normalize, split, blocking, features, models, decide,
                                metric, submission, data, utils
scripts/                        eda_*, measure_blocking, build_pairs, train_model,
                                test_* (run these after touching src/), make_submission_zip
train.py  inference.py          end-to-end entry points
configs/                        one yaml per run
experiments/                    log.md + per-run snapshots
submissions/                    archived copies of every uploaded file (tracked)
docs/                           methodology write-up
package/                        staging area for the final zip — gitignored
notebooks/                      exploration only; anything feeding a submission moves to src/
```
