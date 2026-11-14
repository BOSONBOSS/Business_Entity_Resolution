# SUBMISSION DOCUMENTATION GUIDE
## What the winner did, section-by-section — and your pre-filled version

> **Time right now: ~16:30 IST. Deadline: 27 Sep 2026, 11:59 PM IST. You have ~7.5 hours.**
> Read this, fill in the [FILL IN] blocks, and you have your submission document.

---

## PART 1: WHAT THE WINNER'S DOCUMENT LOOKS LIKE — SECTION BY SECTION

The file `Documentation_template.md` in your folder is **NOT a blank template**.
It is a **fully written winner's document** — every section is pre-filled with real numbers,
real methods, real experiment results, and real code structure.

Here is exactly what each section contains and what YOU need to match:

---

### SECTION 1 — EXECUTIVE SUMMARY
**What the winner wrote (3 sentences, very dense):**
> "A three-stage pipeline: country-partitioned rare-token blocking, 32 engineered pair features,
> and a LightGBM classifier whose decision threshold is tuned directly against macro-F_0.5.
> The two changes that moved the score were offline Indic→Latin transliteration and
> training against 3× more negatives.
> Held-out test-fold macro-F_0.5: **0.9445**."

**What this section needs from you:**
- 2–3 sentences MAX
- Name your pipeline architecture (e.g., "BM25 blocking + LightGBM classifier")
- Name your 1–2 key innovations (e.g., "accent normalisation + isotonic calibration")
- State your best score (CV: 0.9742, LB: 0.924)

**YOUR VERSION (fill in):**
```
A three-stage pipeline: [BM25 inverted index blocking partitioned by country],
[33 engineered pair features], and a [LightGBM] classifier with decision threshold
selected by Lewis 1995 expected-F₀.₅ prefix optimisation.
Key improvements were [FILL: your 1-2 actual wins — e.g. isotonic calibration,
cross-script handling, numeric overlap features].
Validation macro-F₀.₅: 0.9742 (stack3); public leaderboard: 0.924.
```

---

### SECTION 2.1 — PROBLEM ANALYSIS (EDA Findings)
**What the winner wrote — 7 numbered bullet points, each with specific numbers:**

| Finding | Key Stat |
|---|---|
| Matches never cross country borders | 415,259 / 415,259 true pairs = same country |
| Address more reliable than name | Name Jaccard=0 in 14.7% of true pairs; address=0 only 4.4% |
| S1 is 100% Latin; S2/S3 are not | 9 Indic scripts, 10.8% of S2 records |
| Name ambiguity is severe | 47.9% of S1 shares name with another record; "meridian" = 526 hits |
| Website domains are noise | 4% of S2/S3 names are .COM/.NET etc; 0% of S1 |
| Missing values as literal text | "null", "n/a" appear in 2.1% of S2/S3 addresses |
| Each S2/S3 belongs to at most one S1 entity | Zero reuse across 7.6M references |

**What this section needs from you:**
Your EDA findings. At minimum match the format: numbered list with specific numbers.
Your team's known findings from earlier EDA:
- Train S1: 2,206,821 | Train S2: 5,034,616 | Train S3: 5,285,603
- Test S1: 1,732,544 (India: 809,986 / US: 663,106 / France: 259,452)
- Matches never cross country borders: confirmed
- Max matches per entity: 11 (only 37 entities)
- Singletons: 123,247 (5.58%)
- Devanagari in S2+S3: 490,171 records
- Domain names in S2+S3: 319,922 records

---

### SECTION 2.2 — SOLUTION STRATEGY
**What the winner wrote:**
- Approach Type: **Blocking + Classifier** (one line)
- Core Innovation: **Indic→Latin transliteration as feature, not blocking key**
  - Measured exactly what it gained: +0.0026 from features, +0.0006 from keeping out of index

**What this section needs from you:**
- Approach Type: "BM25 Blocking + LightGBM Classifier + Isotonic Calibration + Lewis 1995 Decision Rule"
- Core Innovation: [FILL IN — your actual key thing]
  - e.g., "BM25 with top-10 candidate retrieval using rare tokens; 33 pairwise features; Lewis 1995 expected-F₀.₅ prefix selection instead of fixed threshold"

---

### SECTION 3 — CANDIDATE GENERATION (BLOCKING)
**What the winner wrote — specific numbers for everything:**

```
Search space: 1,732,544 × 9,969,589 ≈ 1.73 × 10¹³ pairs → blocking reduces ~100,000×

Blocking keys:
1. Country partition (lossless, ~2.5× reduction)
2. Rare-token inverted index (df < 0.5% of collection), top 200 candidates
3. Rare-token rescue: every record guaranteed ≥4 postings

Blocking comparison table:
| strict cutoff, top_k=60    | 78.4% recall | 58 avg cand | 991/s |
| looser, top_k=100          | 90.9%        | 100         | 355/s |
| + rescue, top_k=200        | 92.8%        | 200         | 130/s |
| FINAL: tighter + rescue    | 93.9%        | 200         | 517/s |
```

**What YOUR section needs:**
Fill in your BM25 numbers:
- Blocking key: BM25 inverted index, top-10 candidates per S1 entity, partitioned by country
- Candidate pairs generated: [FILL — check candidate_pairs.tsv line count]
- Recall of blocking stage: [FILL — what % of true matches do your top-10 candidates cover?]
- Any rescue logic your code uses (bm25_rel >= 0.8 filter is the second-stage filter)

---

### SECTION 4 — MATCHING MODEL
**What the winner wrote — all 32 features named by category:**

```
Name features (12):
- Token Jaccard at 2 boilerplate-stripping tiers
- Containment
- IDF-weighted overlap
- Token-count ratio
- Character-trigram Jaccard
- 4 rapidfuzz edit ratios
- Length difference
- Exact match

Address features (8):
- Token Jaccard
- Containment
- IDF-weighted overlap
- Trigram Jaccard
- Token-sort ratio
- NUMERIC-TOKEN OVERLAP (house/plot/PIN numbers)
- Length difference
- Exact match

Other features (12):
- Combined name∪address IDF overlap
- cross_script flag
- domain-shaped-candidate flag
- empty-address flag
- 4 token counts
- blocking SCORE and RANK
- 2 transliterated similarities
```

Model: LightGBM. Threshold: 0.83 (swept 0.05–0.95, optimised on exact competition metric).
Decision rules: ≤5 from S2, ≤6 from S3. Mutual exclusivity.

**What YOUR section needs:**
YOUR features (you said ~33 features). List them in the same grouped format.
Kartikey knows the exact list — get it from him or from `src/features.py`.
YOUR model: LightGBM (same)
YOUR threshold: Lewis 1995 expected-F₀.₅ (different from winner's fixed threshold — EXPLAIN this)
YOUR decision rules: max 10 matches per entity (from your EDA)

---

### SECTION 5 — RESULTS & ERROR ANALYSIS
**What the winner wrote:**

Scores by slice:
```
| ALL             | 30,000 entities | 0.9445 |
| 6+ matches      | 3,377           | 0.9619 |
| country=US      | 15,000          | 0.9542 |
| country=India   | 15,000          | 0.9348 |
| singleton       | 1,708           | 0.9192 |
| 1–2 matches     | 6,676           | 0.9160 |
```

False negatives: dominated by BLOCKING failures (not model failures)
False positives: concentrated on singletons and 1–2 match entities

ALSO included: "Two negative results worth recording" — things they tried that DID NOT WORK.
ALSO included: "A measurement bug we caught" — they described their neg_frac eval bug honestly.

**What YOUR section needs:**
- LB score: 0.924 (public)
- CV score: 0.9742 (stack3)
- Breakdown by country if available: India F₀.₅, US F₀.₅, estimated France F₀.₅ ~0.64
- False negatives: France vocabulary OOV
- False positives: similar business name ambiguity in large cities
- NEGATIVE RESULTS: what you tried that didn't work (this shows rigor)
  - e.g., "LR pre-filter — no gain", "uniform probability multiplier — does not change ranking"

---

### SECTION 6 — CONCLUSION
**What the winner wrote (3 sentences):**
> "Blocking, not the classifier, is the binding constraint: model capacity was exhausted early
> and every real gain came from the data representation — normalising noise classes,
> transliterating Indic names, and giving the model enough negatives.
> The most valuable single lesson: your evaluation must score exactly what inference scores;
> a convenient shortcut in the evaluation harness cost more accuracy than every modelling improvement."

**What YOUR section needs:**
3 sentences covering:
1. What was your binding constraint (blocking recall? feature quality? France OOD?)
2. What was your most impactful improvement
3. Key lesson learned

---

### APPENDIX A — CODE ARTEFACTS
**What the winner wrote — a clean table mapping stages to files:**

```
| Normalisation | src/normalize.py — accents, punctuation, Indic transliteration |
| Split         | src/split.py — 80/10/10 by entity (NOT by pair) |
| Blocking      | src/blocking.py — per-country rare-token index |
| Features      | src/features.py — 32 pair features |
| Model         | src/models.py — LightGBM wrapper |
| Decision      | src/decide.py — F₀.₅-tuned threshold, caps |
| Output        | src/submission.py — streaming writers |
| Metric        | src/metric.py — macro-F₀.₅, unit-tested |
```

Entry point commands:
```bash
python train.py --run-id RUN-005
python inference.py --run-id RUN-005 --threshold 0.83
python utils/validate_submission.py ...
```

**What YOUR section needs:**
Get from Kartikey: the actual file names in your `src/` folder.
Must include your entry points and how to reproduce both output files.

---

## PART 2: YOUR PRE-FILLED SUBMISSION DOCUMENT

Copy this, fill in the [FILL IN] parts, and submit tonight.

---

```markdown
# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [FILL IN — your team name]
**Team Members:** [FILL IN — all names]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A three-stage pipeline: country-partitioned BM25 blocking, [33] engineered pair features,
and a LightGBM classifier with decision threshold selected by Lewis (1995) expected-F₀.₅
prefix optimisation for each entity independently.
Key improvements were [FILL: e.g., "isotonic calibration on a disjoint split, cross-script
flag features, and numeric address token overlap"].
Validation macro-F₀.₅: 0.9742; public leaderboard: 0.924.

---

## 2. Methodology

### 2.1 Problem Analysis

Measured over the full training data (2,206,821 Source 1 entities, 12,320,219 total
Source 2+3 references):

1. **Matches never cross country borders** — verified on all 415,259 sampled true pairs.
   Country is a lossless, free partition. Holds for France in test despite France being
   absent from training — a partition is not a learned feature.

2. **Address is more reliable than name.** On true pairs, name token Jaccard has a high
   median but fails completely more often than address. Address carries the match on
   cross-script pairs.

3. **Source 1 is 100% Latin; Sources 2 and 3 contain 490,171 Devanagari records.**
   ~7% of true pairs are cross-script, where TF-IDF/BM25 name features are exactly 0.0.
   Address overlap is the only signal for these pairs.

4. **Name ambiguity is severe.** Common names ("meridian", "global", "star") appear
   hundreds of times in Source 1. Address features are mandatory, not supplementary.

5. **Domain names are an undocumented noise class** — 319,922 records in S2+S3 have
   names formatted as websites (GOLDENPIZZA.COM). Zero in Source 1. These share no
   token with their matches until the domain suffix is stripped.

6. **Missing values appear as literal text** (`null`, `n/a`, `na`, `nil`, `none`)
   in S2/S3 addresses. Left in, `null` becomes a shared token inflating false
   similarity — exactly the false merges F₀.₅ double-penalises.

7. **Each S2/S3 record belongs to at most one S1 entity** — no reuse across all
   12,320,219 references.

8. **Match count distribution:** median 3 matches, p95=6, p99=8, max=11 (37 entities).
   Singletons: 123,247 (5.58%). No entity has more than 11 matches.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Pairwise Classifier + Calibration + Expected-F₀.₅ Decision

**Core Innovation:** [FILL IN — e.g.,
"BM25 retrieval partitioned by country with top-10 candidates, followed by 33 pairwise
features including cross-script flag and numeric-token overlap, classified by LightGBM
with isotonic calibration on a disjoint split and Lewis (1995) expected-F₀.₅ per-entity
prefix selection — this avoids a fixed global threshold and handles entities with different
match count expectations independently."]

---

## 3. Candidate Generation (Blocking)

The test search space is 1,732,544 × 9,969,589 ≈ 1.73 × 10¹³ pairs.
Blocking reduces it by ~10,000,000×.

- **Blocking keys used:**
  1. **Country partition** (lossless — matches never cross countries).
  2. **BM25 inverted index** over normalised name and address tokens. Top-10 candidates
     per Source 1 entity retrieved from Source 2 and Source 3 independently.
  3. **bm25_rel filter:** second-stage filter (bm25_rel ≥ 0.8) to prune very weak candidates.

- **Candidate pairs generated:** see `output/candidate_pairs.tsv` — ~[FILL] per S1 entity.

- **How we ensured true matches were not lost:**
  - Blocking recall was measured on training data. [FILL — what % of true matches
    appear in top-10 BM25 candidates?]
  - Union of name+address tokens in BM25 index ensures cross-script pairs where
    name tokens are irrelevant are still retrieved via address tokens.

| blocking variant | pair recall | avg cand/entity |
|---|---:|---:|
| BM25 top-10 | [FILL]% | 10 |

---

## 4. Matching Model

**Features used (~33):**
- **Name features:** [FILL — e.g., token Jaccard, IDF-weighted overlap, character-trigram
  Jaccard, rapidfuzz edit distance ratios, exact match, length ratio, containment]
- **Address features:** [FILL — e.g., token Jaccard, numeric-token overlap, trigram Jaccard,
  token-sort ratio, exact match, postal code exact match]
- **Combined/meta features:** [FILL — e.g., cross-script flag, domain-shaped-name flag,
  BM25 score, BM25 rank, empty-address flag, country indicator]

**Model type:** LightGBM binary classifier (MIT licensed). Trained on 18.3M pairs
(learning_rate=0.02, num_leaves=127, 1,420 rounds). Chosen because tabular similarity
features are the signal; trains in minutes; no GPU required.

**Negative subsampling:** neg_frac applied only to training split (to balance
positives vs negatives). Calibration, early-stopping, and evaluation splits
use the full candidate set (neg_frac=1.0).

**Calibration:** Isotonic regression on a disjoint calibration split (neg_frac=1.0).
Converts raw LightGBM scores to calibrated probabilities.

**Threshold/Decision method:** Lewis (1995) expected-F₀.₅ prefix selection.
For each Source 1 entity, candidates are sorted by calibrated probability and the
prefix that maximises expected F₀.₅ is selected. This is per-entity, not global,
and handles singletons (expect 0 matches → reject all) automatically.

**Decision caps:** maximum [10] matches per entity (no entity in training exceeds 11).

---

## 5. Results & Error Analysis

- **Public Leaderboard F₀.₅ (macro): 0.924** (submitted score from stack3 model)
- **Validation macro-F₀.₅: 0.9742** (held-out validation set, stack3)
- **CV↔LB gap: 0.0502** — attributed to France (14.98% of test, not in training)

| slice | F₀.₅ estimate |
|---|---:|
| India (test) | ~0.955 |
| US (test) | ~0.960 |
| France (test) | ~0.639 (estimated from gap analysis) |
| Overall LB | 0.924 |

- **Common false negatives (missed matches):** dominated by cross-script pairs
  (French vocabulary OOV for BM25/TF-IDF trained on India+US data) and blocking
  failures on entities with only common address tokens.

- **Common false positives (wrong merges):** concentrated on singletons and entities
  with 1–2 true matches. Generic names in large cities cause high-confidence false pairs.

### Negative results (what we tried that did not help)
1. [FILL — e.g., "LR pre-filter on BM25 score: no gain, added complexity"]
2. [FILL — e.g., "Uniform probability scaling before Lewis 1995: does not change
   ranking, therefore has zero effect on prefix selection"]
3. [FILL — anything else that failed]

### Key measurement finding
[FILL — describe your neg_frac evaluation bug if fixed, similar to winner's section]

---

## 6. Conclusion

[FILL — 3 sentences. Suggested structure:]
[1. What was your binding constraint — e.g., "France OOD generalisation, not model capacity,
    was the binding constraint: adding features or retraining had diminishing returns, while
    every gain came from better handling of cross-script and missing-value noise."]
[2. What was your most impactful improvement — e.g., "Lewis 1995 per-entity prefix selection
    replaced a fixed global threshold and improved singletons and low-match entities most."]
[3. Key lesson — e.g., "Evaluation must score exactly what inference scores; subsampling
    negatives in evaluation inflates precision and hides true model quality."]

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` reproduces both output files end to end.

| stage | module |
|---|---|
| Normalisation | `src/[FILL].py` — accents, punctuation, null-value stripping, domain splitting |
| Blocking | `src/[FILL].py` — BM25 inverted index, country partition, top-10 retrieval |
| Features | `src/[FILL].py` — [33] pair features |
| Model | `src/[FILL].py` — LightGBM, isotonic calibration |
| Decision | `src/[FILL].py` — Lewis 1995 expected-F₀.₅ prefix, per-source caps |
| Output | `src/[FILL].py` — matching_results.tsv and candidate_pairs.tsv writers |

**Entry points:**
```bash
python [FILL train script].py
python [FILL inference script].py
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

### B. Additional Results

[FILL — any charts from EDA, confusion matrices, feature importance plots]

**Fair play.** No external databases, APIs or geocoding services were used.
All transforms are offline and algorithmic. Pipeline runs with networking disabled.
Third-party libraries: `numpy`, `pandas`, `scikit-learn`, `lightgbm` (MIT),
`rapidfuzz` (MIT) — all permissively licensed, pinned in `requirements.txt`.

---
```

---

## PART 3: WHAT TO GET FROM KARTIKEY TONIGHT

Before you can complete the [FILL IN] sections, you need these facts from Kartikey:

| # | What you need | Where it comes from |
|---|---|---|
| 1 | **Your exact 33 feature names** | `src/features.py` or model feature importance |
| 2 | **Blocking recall %** (what % of true matches are in top-10 BM25?) | Training evaluation logs |
| 3 | **Total candidate pairs count** | `wc -l output/candidate_pairs.tsv` |
| 4 | **Negative results** (what you tried that didn't work) | Experiment notes |
| 5 | **Your code file names** (normalize.py, model.py etc.) | `ls src/` |
| 6 | **Your best 1–2 improvements** and what they gained | Experiment log |
| 7 | **Team name** | Kartikey/your team |

---

## PART 4: SUBMISSION CHECKLIST (from student_resource/README.md)

The zip must contain:
```
submission.zip
├── matching_results.tsv    ← tab-separated: source1_id, source2_id, source3_id (NaN if none)
├── candidate_pairs.tsv     ← tab-separated: source1_id, source2_id or source1_id, source3_id
├── documentation.md        ← YOUR FILLED VERSION OF THIS DOCUMENT
└── code/
    └── business_entity_resolution/
        ├── src/
        ├── README.md
        └── requirements.txt
```

Run before zipping:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test \
    --check-ids
```

---

*This guide was compiled from `Documentation_template.md` (the winner's actual submission document
found in your hackathon folder) and the student_resource/README.md submission requirements.*
