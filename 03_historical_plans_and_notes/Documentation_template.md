# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [FILL IN]
**Team Members:** [FILL IN]
**Submission Date:** 2026-09-25

---

## 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*

A three-stage pipeline: country-partitioned rare-token blocking, 32 engineered pair features, and
a LightGBM classifier whose decision threshold is tuned directly against macro-F_0.5. The two
changes that moved the score were **offline Indic→Latin transliteration** (Source 1 is 100% Latin
while ~7% of true matches have a non-Latin name, so every name feature was otherwise exactly zero
on that slice) and **training against 3× more negatives**, which sharpened the model precisely at
the high-precision operating point the metric forces us into. Held-out test-fold macro-F_0.5:
**0.9445**.

---

## 2. Methodology

### 2.1 Problem Analysis
*Key insights discovered during EDA — noise patterns, address variations, missing fields, etc.*

Measured over the full training data (2.2M Source 1 entities, 7.6M ground-truth references):

1. **Matches never cross country borders** — 415,259 of 415,259 sampled true pairs are
   same-country. Country is therefore a free, lossless partition, and it holds for France despite
   France being absent from training, because a partition is not a learned feature.
2. **The address is more reliable than the name.** On true pairs, name token Jaccard has a median
   of 1.000 but is **zero 14.7%** of the time; the address has a lower median (0.625) but is zero
   only **4.4%** of the time. The name matches perfectly more often and fails completely three
   times as often.
3. **Source 1 is 100% Latin; Sources 2 and 3 are not.** Nine Indic scripts appear (Devanagari,
   Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya, Gurmukhi) in 10.8% and 5.5% of
   S2/S3 records. So ~7% of true pairs are Latin↔Indic, where name similarity is *exactly* 0 and
   the address alone carries the match — and on those pairs the address behaves unusually well
   (median Jaccard 0.700, zero-overlap just 0.14%).
4. **Name ambiguity is severe.** 47.9% of Source 1 records share a normalised name with another
   record; `meridian` appears 526 times. Nearly half the dataset cannot be identified by name
   alone, which makes the address mandatory rather than supplementary.
5. **Website domains are an undocumented noise class** — 4% of S2/S3 names, **0.00%** of Source 1.
   `GOLDENPIZZA.COM` vs `Golden Pizza` shares no token until the TLD is stripped and the stem
   re-segmented; 32% of all zero-token-overlap true pairs are this.
6. **Missing values appear as literal text** (`null`, `n/a`, `na`, `nil`) in ~2.1% of S2/S3
   addresses, on top of ~2.7% blank. Left in, `null` becomes a token two unrelated addresses
   *share*, inflating similarity and manufacturing exactly the false merges F_0.5 punishes most.
7. **Each S2/S3 record belongs to at most one S1 entity** — zero reuse across all 7,638,365
   references.

### 2.2 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** Blocking + Classifier
**Core Innovation:** Offline phonetic Indic→Latin transliteration used as a **feature**, not as a
blocking key. Source 1 is entirely Latin, so for the ~7% of cross-script true pairs every name
feature is exactly 0.0 and the model must decide on the address alone. Transliterating lifts those
pairs from **0/8 to 8/8** with non-zero name similarity. Critically, we measured that indexing the
transliterated tokens *hurts* (it attaches Indic records to already-common English tokens,
diluting the IDF ranking and costing both recall and ~20% throughput), so transliteration is
confined to the feature stage: **+0.0026 from the features, +0.0006 more from keeping it out of
the index.**

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

The test search space is 1,732,544 × 9,969,589 ≈ **1.73 × 10¹³ pairs** — about 75 hours for a
single string metric even across 32 threads. Blocking reduces it ~100,000×.

- **Blocking keys used:**
  1. **Country partition** (lossless, ~2.5× reduction).
  2. **Rare-token inverted index** over the union of normalised **name and address** tokens,
     document frequency below 0.5% of the collection. Candidates are ranked by summed IDF of
     shared tokens, so one rare token ("madhulata") outweighs several common ones ("new", "road",
     "delhi"). Top 200 per entity retained.
  3. **Rare-token rescue:** every record is guaranteed ≥4 postings using its *own* rarest tokens
     even when they exceed the global cutoff.
- **Candidate pairs generated:** see `output/candidate_pairs.tsv` — ~200 per Source 1 entity,
  ≈ 3.4 × 10⁸ in total (reduction ratio ≈ 2 × 10⁻⁵ of the full cross product).
- **How we ensured true matches were not lost:**
  - Recall was **measured every run** and treated as a first-class metric, separating
    *reachability* (does the index connect the pair at all) from *truncation* (retrieved but
    ranked below `top_k`) — they have different fixes, and that distinction drove every
    improvement below.
  - **Name-only blocking reaches just 85.1% and address-only 95.6%; the union reaches 99.8%
    reachability.** Name-only would have capped the final score before the model ever ran.
  - The rescue exists because a record whose every token is common ("SHOP NO 5, MAIN ROAD,
    DELHI") is otherwise absent from the index and unreachable at any `top_k` — measured at 3.55%
    of true matches.

| blocking variant | pair recall | avg cand/entity | throughput |
|---|---:|---:|---:|
| strict cutoff, top_k=60 | 78.4% | 58 | 991/s |
| looser cutoff + vectorised query, top_k=100 | 90.9% | 100 | 355/s |
| + rare-token rescue, top_k=200 | 92.8% | 200 | 130/s |
| **final: tighter cutoff + rescue, top_k=200** | **93.9%** | **200** | **517/s** |
| reachability ceiling | 97.7% | — | — |

The final configuration is better on *both* axes than the looser-cutoff variant. Counter-
intuitively, **tightening the df cutoff and rescuing per record beats loosening the cutoff**: a
loose cutoff mostly adds very long postings for tokens that were never going to disambiguate
anything.

---

## 4. Matching Model

**Features used (32):**
- **Name (12):** token Jaccard at two boilerplate-stripping tiers, containment, IDF-weighted
  overlap, token-count ratio, character-trigram Jaccard, four rapidfuzz edit ratios, length
  difference, exact match.
- **Address (8):** token Jaccard, containment, IDF-weighted overlap, trigram Jaccard,
  token-sort ratio, **numeric-token overlap** (house/plot/PIN numbers are highly discriminative),
  length difference, exact match.
- **Other (12):** combined name∪address IDF overlap, `cross_script` flag, domain-shaped-candidate
  flag, empty-address flag, four token counts, blocking **score** and **rank**, and two
  **transliterated** similarities (trigram Jaccard and token-sort ratio on the Latin forms).

**Model type:** LightGBM binary classifier (MIT licensed; gradient-boosted trees, far under the
8B-parameter limit). Chosen because the signal is tabular similarity, it trains in minutes on CPU
across 32 threads — so several full iterations fit inside the 3-day window — and it needs no GPU.

**Threshold selection method:** F_0.5 optimisation on the held-out validation split, swept
0.05–0.95 and scored with the exact competition metric. **Optimum 0.83**, far above 0.5, which is
what the metric's 2× precision weighting predicts.

**Decision rules:** per-source caps (≤5 from S2, ≤6 from S3 — no entity among 2.2M ever exceeds
these) and mutual exclusivity on the S2/S3 side.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro): 0.9445** on a held-out test fold (30,000 entities), threshold 0.83
  applied unchanged from validation (which scored 0.9437 — it transfers).
  Micro precision **0.9808**, micro recall **0.8902**.

| slice | n | macro-F_0.5 |
|---|---:|---:|
| ALL | 30,000 | **0.9445** |
| 6+ matches | 3,377 | 0.9619 |
| country=US | 15,000 | 0.9542 |
| 3–5 matches | 18,239 | 0.9540 |
| country=India | 15,000 | 0.9348 |
| singleton | 1,708 | 0.9192 |
| 1–2 matches | 6,676 | 0.9160 |

- **Common false negatives (missed matches):** dominated by **blocking, not the model**. Candidate
  recall is 93.9% and the pipeline achieves 89.0% recall at the precision-optimal threshold, so
  the classifier extracts most of what retrieval supplies. The residual splits into ~2.3% of true
  pairs sharing **no token at all** after normalisation (cross-script name *and* differently
  written address) and ~4% retrieved but ranked below `top_k=200`.
- **Common false positives (wrong merges):** concentrated on entities with **few or no true
  matches**. Singletons (0.9192) and 1–2-match entities (0.9160) score well below 6+-match
  entities (0.9619). This is structural: with one true match a single false positive halves that
  entity's precision, while with six it barely registers — and the metric macro-averages, so those
  entities count equally. Generic names drive it (`meridian` appears 526 times in Source 1), which
  is why the address features carry more gain than the name features.

### Two negative results worth recording

1. **Mutual exclusivity on the S2/S3 side is worth +0.0000.** The structural fact is real — zero
   record reuse in 7.6M references — but at threshold 0.83 only 3 of 96,702 accepted pairs are
   contested. Once the classifier reaches ~98% precision it stops producing the duplicate claims
   the constraint exists to remove. Retained anyway: free, and never removed a true match.
2. **Context-aware decision rules add nothing.** Relative-to-best and require-a-strong-leader
   rules gained +0.0002 (noise). The gain on singletons (+0.0036) is cancelled by the loss on
   1-match entities (−0.0009), because those two cases are indistinguishable except in whether
   one high-scoring candidate happens to be correct.

### A measurement bug we caught, and what it taught us

Subsampling negatives for *evaluation* (scoring ~43 of each entity's ~200 candidates) inflated
macro-F_0.5 by **0.0368** — more than every genuine improvement combined. Recall was unaffected;
the whole loss was precision, which F_0.5 double-weights. It hit singletons hardest (0.9514 →
0.7881), since an entity with no true matches is scored purely on rejection.

Fixing it mattered twice: it replaced a fictional 0.9639 with a real 0.9417, **and** it exposed a
genuine improvement the flawed metric had hidden. Training with 3× more negatives had looked like
+0.0006 noise under the biased measurement; measured honestly it is **+0.0028**, and the gain is
concentrated on singletons (+0.0246) — exactly the mechanism predicted, since rejection is what
extra negatives teach.

---

## 6. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

Blocking, not the classifier, is the binding constraint: model capacity was exhausted early
(doubling it changed nothing) and every real gain came from the data representation — normalising
the noise classes the EDA exposed, transliterating Indic names into the same space as their Latin
counterparts, and giving the model enough negatives to discriminate at the operating point the
metric forces. The most valuable single lesson was that **your evaluation must score exactly what
inference scores**; a convenient shortcut in the evaluation harness cost more accuracy than every
modelling improvement we made, and hid a real one.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` reproduces both output files end to end with no network access.

| stage | module |
|---|---|
| Normalisation | `src/normalize.py` — accents, punctuation, two-tier legal suffixes, placeholder strings, domain splitting, 9-script Indic transliteration |
| Split | `src/split.py` — deterministic 80/10/10 **by entity** (splitting by pair would leak) |
| Blocking | `src/blocking.py` — per-country rare-token inverted index, rescue, threaded retrieval |
| Features | `src/features.py` — 32 pair features, row-wise and batched paths (verified identical) |
| Model | `src/models.py` — LightGBM wrapper |
| Decision | `src/decide.py` — F_0.5-tuned threshold, mutual exclusivity, per-source caps |
| Output | `src/submission.py` — streaming writers enforcing every format rule |
| Metric | `src/metric.py` — macro-F_0.5, unit-tested against the spec's worked example (0.714) |

**Entry points:**
```bash
python train.py     --run-id RUN-005     # build pairs, fit, tune threshold
python inference.py --run-id RUN-005 --threshold 0.83
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

### B. Additional Results

Full experiment log with every run, config diff and negative result is in `experiments/log.md`.

**Fair play.** No external databases, APIs or services were used to look up or resolve business
identities; no geocoding; no internet data augmentation. Every transform is offline and
algorithmic, and the pipeline runs with networking disabled. Third-party libraries: `numpy`,
`pandas`, `scikit-learn`, `lightgbm` (MIT), `rapidfuzz` (MIT), `indic-transliteration` (MIT) —
all permissively licensed and pinned in `requirements.txt`.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
