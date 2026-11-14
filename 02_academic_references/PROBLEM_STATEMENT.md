# Business Entity Resolution Challenge — Full Spec & Hard Rules

Source: `amazon_ml_challenge_problem_statement.pdf` (8 pages; page 8 blank, no images/annotations).
Explainer video: `https://d8it4huxumps7.cloudfront.net/files/6ab509c5b7036_ml_challenge_2026_video.mp4`

Companion files: [HACKATHON_RULES.md](HACKATHON_RULES.md) (organiser logistics) · [CLAUDE.md](CLAUDE.md) (coding rules).
**Where the two PDFs conflict, see section 12.**

---

## 1. The task

Given business records from **3 independent sources** with noisy, inconsistent fields and **no
shared identifiers**, determine which records refer to the same real-world business.

- **Source 1 is the deduplicated reference source.**
- For **each Source 1 entity**, find all matching records from **Source 2 and Source 3**.
- A Source 1 entity may match **zero, one, or many** records from S2 and S3.
- A record's source is given by its `entity_id` prefix (`S1-`/`S2-`/`S3-`) and by which file it is
  in. **There is no `source` column.**

## 2. File format — TSV, always

All files are **tab-separated** (`.tsv`), and **submissions must be tab-separated too**. Tabs are
used because business addresses *and* the ID-list columns contain commas.

```python
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```

> Reading a `.tsv` without `sep="\t"` **silently** produces a single column containing the whole
> line. Every read and every write in this repo passes an explicit separator. No exceptions.

## 3. Input schema

Each `*_source1.tsv` / `*_source2.tsv` / `*_source3.tsv`:

| Column | Description |
|---|---|
| `entity_id` | Unique record id. Prefix `S1-` / `S2-` / `S3-` indicates the source. |
| `business_name` | May contain abbreviations, legal suffixes, typos, transliterations. |
| `business_address` | May be partial, format-varied, missing components, landmark-based. |
| `country` | Country label. **Open set of string labels.** |

`train_ground_truth.tsv`:

| Column | Description |
|---|---|
| `source1_entity_id` | `entity_id` of a Source 1 record |
| `matched_entity_ids` | Comma-separated matching ids from S2 and/or S3; **empty when no matches** |

### 3.1 The country trap — read twice

Training data covers **US and India**. The test set contains a **third country, France, that does
not appear in training data**.

- **Do not hard-code, filter, or one-hot the pipeline to `{US, India}`.** Treat `country` as an
  open set of string labels.
- **Every test entity — France included — must appear in the submission.**
- Any country-conditioned logic must have a working default branch for unseen labels.
- France means unseen address conventions and unseen name conventions. Anything learned
  US/India-specific must degrade gracefully, not crash or silently drop rows.

## 4. Noise patterns to expect

**Name variations:** abbreviations (Corp↔Corporation, Pvt↔Private, Ltd↔Limited), legal-suffix
inconsistencies, DBA/trade names, punctuation (`&` ↔ "and"), word-order transpositions, typos.

**Address variations:** abbreviations (Rd↔Road, St↔Street), transliteration variants, missing
components (no PIN code, no state), landmark-based references ("Near SBI ATM"), municipal
numbering formats, component reordering.

## 5. Files on disk

```
dataset/train/train_source1.tsv       # Source 1 training records (deduplicated reference)
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv  # labels
dataset/test/test_source1.tsv         # generate matches for EVERY entity in this file
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

**No ground truth for the test set.** Hold out a validation split from training data and score it
locally with the F_0.5 formula in section 8.

## 6. Output — two files, both in `output/`

### 6.1 `matching_results.tsv` — the only leaderboard-scored file

| Column | Description |
|---|---|
| `source1_entity_id` | `entity_id` of a Source 1 record |
| `matched_entity_ids` | Comma-separated matching ids from S2 and/or S3 |

Columns separated by a **single tab**; ID lists separated by **commas with no quoting**.

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

**Rules (violation = rejection, not a low score):**
- Every Source 1 entity in the test set has **exactly one row**
- `matched_entity_ids` **empty** for singletons
- **No duplicate entity IDs within a single ID list**
- **No duplicate `source1_entity_id` rows**
- ID lists contain **only S2-/S3- IDs that exist in the test set**
- **No self-matches to Source 1** — rejected

### 6.2 `candidate_pairs.tsv` — the blocking set

| Column | Description |
|---|---|
| `source1_entity_id` | `entity_id` of a Source 1 record |
| `candidate_entity_ids` | Comma-separated candidate ids from S2 and/or S3 |

**Definition, precisely:** the **exact set of records fed into the matching model for inference** —
the final candidate list just before the ML model scores them. **Not** the raw output of an early
blocking pass that is later filtered further. If the pipeline has several blocking/filtering
stages, this is the **last** one: whatever the model actually runs inference over.

- Same formatting rules as `matching_results.tsv` (one row per S1 entity, empty when no candidates,
  S2-/S3- only, no duplicates within a list).
- **Every ID in `matching_results.tsv` must appear here.** Final matches must be a **subset** of
  candidates — a matched ID that never appeared as a candidate is a pipeline bug and the validator
  warns about it.
- **Not scored on the leaderboard.** Used to analyse blocking quality (recall ceiling, reduction
  ratio) and to verify the pipeline.

### 6.3 Validate before every submission

The bundle ships `utils/validate_submission.py` (stdlib only, Python 3.8+). It is present at
`utils/validate_submission.py` in this repo. Run from the repo root:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Prints `PASS` (exit 0) or a numbered list of issues (exit 1). It does not compute the score.
Running it costs nothing; a rejected upload costs a submission slot. **Never upload without a PASS.**

#### What the validator source actually enforces (read from the script, not the PDF)

**Hard errors (exit 1):**
- Header must equal `['source1_entity_id', 'matched_entity_ids']` exactly — compared after
  `.strip().lower()`, so case and surrounding whitespace are forgiven, but nothing else is.
- A header with no tab but containing commas is caught explicitly as "you submitted a CSV".
- Duplicate `source1_entity_id` rows.
- Duplicate ids **within** one list.
- Any `S1-` id in a match list (self-match).
- Any id without an `S2-`/`S3-` prefix.
- **Missing** required S1 entities *and* **extra** S1 ids not in the test set — both directions.
- A non-blank row containing **no tab at all** is "malformed". A singleton row must therefore be
  `S1-00003<TAB>` with the trailing tab present, not a bare id. `df.to_csv(sep='\t')` does this
  correctly; hand-rolled writers often do not.
- Non-UTF-8 content fails with a dedicated message.

**Warnings only (still exit 0):**
- `candidate_pairs.tsv` missing — it is *optional to the validator*, though still required in the
  final zip.
- Matched ids not present in `candidate_pairs.tsv` — the subset rule is a **warning, never a
  failure**.
- The ID-existence check being off.

#### ⚠️ Two places the validator contradicts the PDF

1. **`--check-ids` is OFF by default.** The default run does **not** verify that matched ids exist
   in the test set. Pass `--check-ids` to enable it (it loads all S2/S3 ids — a few GB of RAM).
   A default `PASS` therefore does **not** prove your ids are real.
2. The validator's own docstring says a missing or garbage matched id **"only lowers your score
   rather than being rejected by the scorer"** — whereas the PDF says such ids "will be rejected".
   **Follow the stricter PDF reading**: treat nonexistent ids as unacceptable. But do not rely on a
   default validator run to catch them — run `--check-ids` at least once before the final submission.

Other notes from the source: ids are **not** stripped when splitting on commas, so a space after a
comma creates an id like `" S2-00047"` that fails the prefix check — write lists with **no spaces**.
Files are opened in text mode, so Windows CRLF line endings are normalised on read and are safe
(verified empirically on this machine with pandas 2.3.3).

## 7. Final submission package

A single zip, in addition to live leaderboard uploads. **All teams must submit it.** Top teams'
packages are reviewed in detail before final rankings are confirmed — used to reproduce results,
audit blocking, and check fair-play and model-license rules.

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (same file uploaded to the leaderboard)
│   └── candidate_pairs.tsv         # blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all source code
│       ├── README.md               # how to reproduce end-to-end (data → blocking → matching → output)
│       └── requirements.txt        # pinned dependencies / environment
└── Documentation_template.md       # filled-in methodology write-up
```

- `code/business_entity_resolution/` must be **self-contained and runnable**. Anyone must be able
  to regenerate **both** output files from the training/test data using only what is in that folder.
- `requirements.txt` must **pin versions**.
- Drop the filled-in `Documentation_template.md` straight in — **no need to rename it**. The filled
  `.md` is fine; a `.pdf` export also works.

## 8. Evaluation — F_0.5, macro-averaged

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- **β = 0.5 → precision-heavy.** Weights precision **2× over recall**. False merges (matching two
  different businesses) are penalised more than missed matches.
- **Computed per Source 1 entity, then macro-averaged** across all Source 1 entities in the
  evaluation set. Every entity counts equally regardless of how many matches it has.
- **Singletons are included in the average.** An entity with no true matches scores **1.0** when an
  empty list is correctly predicted, and **0.0** when any match is predicted for it.

Worked example from the spec:
- Predicted `S1-00001` → `[S2-00047, S2-00193, S3-00812]`; truth → `[S2-00047, S3-00812]`
- Precision = 2/3, Recall = 2/2 = 1.0
- F_0.5 = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) = **0.714**

**Strategic consequences:**
- A macro-average over entities means one bad prediction on a singleton costs as much as one bad
  prediction on a 10-match entity. **Singletons are free points — do not neglect them.**
- Adding a wrong match to a correct prediction hurts more than omitting a right one. When the
  model is uncertain, **predicting nothing beats guessing**.
- Threshold tuning is the single highest-leverage knob. Tune it on held-out data against this exact
  macro-F_0.5, not against pair-level accuracy or AUC.

## 9. Leaderboards

- **Public:** during the challenge, a **subset** of the test set — real-time feedback.
- **Private:** after the challenge, the **remaining portion** of the test set.
- **Final rankings are decided by the PRIVATE leaderboard.**
- Predictions are submitted for the **full test set** in both cases; the split is applied at scoring.

## 10. Hard constraints — violating any of these is fatal

1. **Output format exactly as specified.** Submissions failing validation **will not be evaluated**.
   A correctly formatted output shows a **SCORED** status with the F_0.5 score.
2. `matched_entity_ids` references **only Source 2 or Source 3** entities. Self-matches to Source 1
   and IDs not present in the test set are **rejected**.
3. **Every Source 1 entity must appear** in the submission. Missing entities cause **rejection**.
4. **No duplicate entity IDs in any ID list**, and **no duplicate `source1_entity_id` rows** —
   either causes **rejection**.
5. **Final model must be MIT / Apache-2.0 licensed and ≤ 8 billion parameters.**

### 10.1 On the model-licence rule (read before choosing any pretrained model)

Check the licence *before* downloading weights, and record it in the write-up.
- **Safe by licence:** most `sentence-transformers` models (Apache-2.0), BERT/DistilBERT
  (Apache-2.0), E5 / BGE / GTE families (MIT or Apache-2.0), XLM-R (MIT), Qwen (Apache-2.0 for
  most sizes), `rapidfuzz`/`scikit-learn`/`LightGBM`/`XGBoost` (permissive).
- **Not safe:** anything under a bespoke community licence (e.g. Llama licences), non-commercial
  (CC-BY-NC), or research-only weights — regardless of parameter count.
- **≤ 8B parameters** applies to the final model. Verify the parameter count, don't assume it.
- Multilingual capability matters here: test data includes **France** plus transliterated Indian
  addresses. Prefer a permissively licensed multilingual encoder over an English-only one.

## 11. Academic integrity — instant disqualification

> ⚠️ **STRICTLY PROHIBITED: External Data Lookup**

Participants are **strictly not allowed** to use external databases, APIs, or services to look up
business identities or resolve entities. Explicitly named:
- Commercial entity-resolution APIs or services
- Looking up business registrations from government databases
- **Using geocoding APIs to normalise addresses**
- **Any external data augmentation from internet sources**

**Enforcement:** all submitted approaches, methodologies, and code pipelines are thoroughly reviewed
and verified. **Any evidence of external data lookup results in immediate disqualification.**

**What this forbids in practice — check every library before adding it:**
- ❌ Nominatim / OpenStreetMap / Google Maps / HERE / Mapbox geocoding, `geopy` — any network geocoder
- ❌ Scraping, web search, or any runtime network call that enriches a record
- ❌ Downloading business registries, postal-code databases, or company-name gazetteers
- ✅ Pretrained language models used as encoders (subject to the licence + 8B rule in section 10)
- ✅ Offline string/phonetic libraries (`rapidfuzz`, `jellyfish`, `unidecode`, `phonetics`) — they
  are algorithms, not lookups
- ⚠️ Any static list bundled inside a package (e.g. a library's built-in address-abbreviation
  dictionary) is a grey area — prefer rules derived from the training data, and document anything
  bundled that you rely on.

**The pipeline must run with networking disabled.** That is the test to apply before submitting.

## 12. ⚠️ Conflict between the two PDFs — methodology document length

| Source | Says |
|---|---|
| Guidelines PDF | "**1-2 page** document explaining the ML approach, ML models used, experiments and conclusion" |
| Problem statement PDF | Fill in `Documentation_template.md`; "**There is no page limit** — prioritise clarity and technical depth over brevity" |

**Resolution — revised after reading the actual template.** `Documentation_template.md` settles it:
its own section 1 asks for a **"brief 2-3 sentence overview"**, and section 6 for a conclusion "in
2-3 sentences". So the template is *deliberately* concise at the top and detailed in the middle —
it is not a 1–2 page document, nor an unbounded one.

**Follow the template as written.** Do not pad the executive summary to a page; do not compress the
technical sections. The guidelines PDF's "1-2 page" line is satisfied in spirit by the template's
overall brevity, and the problem statement is the challenge-specific spec that names the exact file
the zip must contain. The template closes with: *"Teams can modify sections according to their
approach while maintaining clarity and technical depth"* — so extending a section is permitted,
but there is no reason to restructure it.

Whatever else changes, these four must be covered:

1. Methodology used
2. **Candidate generation / blocking strategy**
3. Model architecture and feature engineering
4. Any other relevant information about the approach

### 12.1 What `Documentation_template.md` actually asks for

The file is at the repo root. Its required fields, so nothing is discovered late:

| Section | Asks for |
|---|---|
| Header | Team Name, Team Members, Submission Date |
| 1. Executive Summary | **2–3 sentences**: approach + key innovations |
| 2.1 Problem Analysis | Key EDA insights — noise patterns, address variations, missing fields |
| 2.2 Solution Strategy | **Approach Type** (Blocking + Classifier / End-to-End / Graph-Based / Hybrid) and **Core Innovation** |
| 3. Candidate Generation | Blocking keys used · **"Candidate pairs generated: [total]"** · how true matches were not lost |
| 4. Matching Model | Name features · Address features · Other · **Model type** · **Threshold selection method** |
| 5. Results & Error Analysis | **F_0.5 (macro)** · common false positives · common false negatives |
| 6. Conclusion | **2–3 sentences** |
| Appendix A | Structure of `code/business_entity_resolution/` and the entry points that regenerate both output files |
| Appendix B | Additional charts / detailed results |

**Two numbers must be tracked from day 1 or they cannot be reported honestly at the end:**
- the **total count of candidate pairs generated** (section 3), and
- the **macro F_0.5 on validation** (section 5).

Section 5 also requires **error analysis** — concrete descriptions of what the model wrongly merged
and what it missed. That means sampling and eyeballing errors during development, not at the
deadline. Keep examples as you find them.

## 13. Organiser tips (verbatim intent)

- **Invest in a strong blocking / candidate generation strategy — it determines the upper bound of
  your recall.** Recall lost at blocking can never be recovered downstream.
- Explore string similarity features: **Jaccard, Levenshtein, TF-IDF cosine** for name and address.
- **Pay attention to country-specific address patterns.**
- Consider the precision-recall trade-off carefully — **F_0.5 rewards precision more than recall**.
- **Do not neglect singletons** — correctly predicting "no match" is worth a full 1.0 on that entity.
- Validate output format against the rules before submitting.

## 14. Derived working checklist

**Before writing pipeline code:**
- [ ] Download the `student_resource` bundle; confirm `utils/validate_submission.py` and
      `Documentation_template.md` are present — **read the validator's source**, it is the exact
      definition of a valid submission
- [ ] Place data at `dataset/train/` and `dataset/test/` (paths mirror the spec verbatim)
- [ ] Implement macro-F_0.5 **exactly** as section 8 defines it, including the singleton rule, and
      unit-test it against the worked example (expected **0.714**)
- [ ] Build the held-out validation split from training data; fix it once

**Before every leaderboard upload:**
- [ ] `utils/validate_submission.py` returns **PASS** (and `--check-ids` passed at least once)
- [ ] Singleton rows end with a trailing tab, not a bare id
- [ ] No spaces after commas in any id list
- [ ] Row count equals `test_source1.tsv` row count exactly
- [ ] Every predicted ID exists in `test_source2.tsv` ∪ `test_source3.tsv`
- [ ] `matching_results.tsv` IDs ⊆ `candidate_pairs.tsv` IDs
- [ ] France/unseen-country entities present and not silently dropped
- [ ] Written to TSV with `sep="\t"`, no index column, no quoting of ID lists
- [ ] Archived to `submissions/` with its config + commit; slot logged in `experiments/log.md`

**Before the final zip:**
- [ ] `code/business_entity_resolution/` regenerates both outputs from scratch, networking off
- [ ] `requirements.txt` pins versions
- [ ] Model licence (MIT/Apache-2.0) and parameter count (≤8B) documented
- [ ] `Documentation_template.md` filled, covering all four required sections
