# ML Course Project — Pre-Submission Evaluation Document
### CSE2032 — Machine Learning | BML Munjal University | 2026–27

---

| Field | Details |
|---|---|
| **Project Title** | Business Entity Resolution across Noisy Multi-Source Records |
| **Course** | CSE2032 — Machine Learning (Credits: 3) |
| **Evaluation Date** | 29 September 2026 |
| **Team** | *(add names)* |
| **Origin** | Amazon ML Challenge 2026 — adapted as End-Semester Project |

---

## 1. Idea / What Are We Doing? (Usme idea kya hai)

### 1.1 The Real-World Problem

Businesses appear in many databases — government registries, delivery platforms, payment systems — but they never share a common identifier. A pizza restaurant might be listed as "Pizza Palace Pvt. Ltd." in one database and "Pizza Palace" in another, with slightly different addresses. **The goal of our project is to automatically determine which records across three independent databases refer to the same real-world business.**

This is called **Business Entity Resolution** (also known as *record linkage* or *entity matching*), and it is one of the core unsolved problems in data engineering, e-commerce, and finance.

### 1.2 Why This Is an ML Problem

The matching cannot be done with a simple exact-string lookup because:

- **Name variations**: `Corp ↔ Corporation`, `Pvt ↔ Private`, `& ↔ and`, word-order transpositions, typos
- **Address variations**: `Rd ↔ Road`, landmark-based references ("Near SBI ATM"), missing PIN codes, component reordering
- **Cross-script names**: A business called "युनिक मार्केटिंग" in Devanagari is the same as "Unique Marketing" in Latin script
- **No shared ID**: The three databases have no common key — only raw text fields (name, address, country)

A human can figure it out — but we have **24 million+ records** across 6 files. That requires an automated ML pipeline.

### 1.3 Our Specific Setup

| File | Records | Countries |
|---|---|---|
| Source 1 (Reference) | 2,206,821 | US, India |
| Source 2 | 5,034,616 | US, India |
| Source 3 | 5,285,603 | US, India |
| **Test Source 1** | 1,732,544 | **US, India, France** |
| Test Source 2 | 4,887,273 | US, India, France |
| Test Source 3 | 5,082,316 | US, India, France |

**The challenge within the challenge**: France appears only in test data — never in training. Our model must generalise to an unseen language/country at inference time. This makes it a genuine out-of-distribution (OOD) generalisation problem.

### 1.4 Output Format

For every Source 1 entity, we predict which Source 2 and Source 3 entities refer to the same business. Example:

```
source1_entity_id    matched_entity_ids
S1-00001             S2-00047,S3-00812
S1-00002             S3-00004
S1-00003             (empty — this is a "singleton", no match)
```

### 1.5 Evaluation Metric: F₀.₅ (Precision-Weighted)

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

This metric penalises **false merges more than missed matches** (precision is weighted 2× over recall). It is macro-averaged — every Source 1 entity counts equally, regardless of how many matches it has. A singleton predicted correctly scores **1.0**; any wrong match on a singleton scores **0.0**.

**Strategic consequence**: Predicting "no match" when uncertain is better than guessing.

---

## 2. Why This Problem — Connection to Our Course (CSE2032)

This project is a direct application of nearly every topic in our ML handout:

| Course Topic | Application in Our Project |
|---|---|
| **Classification** | Is this record-pair a match? (Binary classification) |
| **Feature Engineering** | 38+ handcrafted similarity features from text fields |
| **Ensemble / Boosting** | LightGBM (gradient boosted trees) as the classifier |
| **Unsupervised / Clustering** | Blocking stage — grouping candidate pairs |
| **Model Validation** | Macro F₀.₅, Precision, Recall, held-out validation split |
| **Bias-Variance Tradeoff** | Overfitting to India/US; generalising to France |
| **Hyperparameter Tuning** | Threshold selection, `num_leaves`, `learning_rate` |
| **Dimensionality / PCA** | TF-IDF sparse vectors; vocabulary compression |
| **Bayesian / Probabilistic** | Isotonic calibration of raw model scores → probabilities |
| **Neural Networks (intro)** | Transformer-based cross-encoder (mDeBERTa, XLM-R) |
| **Model Deployment** | Pipeline runs end-to-end from raw TSV → submission file |

This is not a textbook exercise — it is a real competition problem with a leaderboard and 24 million records.

---

## 3. Feasibility

### 3.1 Why This Is Feasible for a Student Team

| Concern | Why It's Manageable |
|---|---|
| **Data is provided** | All 6 TSV files and ground truth are in our repository |
| **Compute** | Runs on a laptop (8 GB RAM for baseline); GPU helps but isn't required for core pipeline |
| **Libraries** | `scikit-learn`, `LightGBM`, `rapidfuzz`, `pandas` — all in our course stack |
| **Metric is defined** | We can compute F₀.₅ locally; no need to upload to get a score |
| **Modular pipeline** | Each stage (blocking → features → model → threshold) is independent |

### 3.2 Complexity Reduction: The Two-Stage Pipeline

The naive approach — comparing all records against all others — involves **17.27 trillion comparisons** (test set alone). This is computationally impossible.

Our solution: a **two-stage approach**:

1. **Blocking (Candidate Generation)**: Use BM25 text retrieval to narrow each query to at most 10 candidate S1 entities. This reduces 17 trillion comparisons to ~17 million (**1,000,000× reduction**).
2. **Matching (ML Classifier)**: Run LightGBM on only those 17 million candidate pairs.

This is a standard pattern in industry-scale entity resolution and is directly implementable with our course tools.

### 3.3 Ground Truth Distribution (What We Learned from EDA)

| True matches per S1 entity | % of entities |
|---|---|
| 0 (singleton) | 5.58% |
| 1 | 5.40% |
| 2 | 17.00% |
| 3–5 | 60.63% |
| 6+ | 11.19% |

Mean matches per entity: ~3.46. Some entities have up to 11 matches.

---

## 4. What We Have Done So Far (Abhi Tak Kya Kiya Hai)

### 4.1 Exploratory Data Analysis (EDA) ✅

We have completed a thorough EDA on all 6 source files. Key findings:

**Scale:**
- Train: 2.2M S1 + 5.0M S2 + 5.3M S3 records; 7.64M labelled true pairs
- Test: 1.73M S1 + 4.89M S2 + 5.08M S3 records
- Total across all files: ~24.2 million records

**Missing data patterns:**
- 610,389 empty addresses across probe files (S2/S3) — ~3–4% of records have no address
- S1 (reference) has **zero** blank addresses — all reference entities are well-formed
- 132 business names with placeholder values ("NA", "NULL") — not truly blank, need special handling

**Country distribution shift (critical finding):**
- Training data: India and US only
- Test data: India (46.7%), France (15.0%), US (38.3%)
- France is completely **absent from training** — zero labelled pairs for French entities

**Name noise patterns observed:**
- Cross-script Devanagari↔Latin names in ~18.7% of sampled true pairs
- Zero word-token overlap in names for ~14.5% of true pairs (yet they are true matches)
- Zero address-token overlap for ~4.3% of true pairs
- Only 0.01% of true pairs have zero overlap in BOTH name AND address simultaneously — we always have at least one signal

**Blocking analysis:**
- Pure name-based BM25 blocker achieves ~97.8% recall @10 candidates per query
- Adding a separate address field index improves recall in the address-missing subset from 90.0% to 95.3%
- Cross-script pairs are a major source of blocking misses (~37% of failures)
- Missing-address pairs account for ~44% of blocking failures

### 4.2 Architecture Design ✅

We have designed the full 6-stage pipeline (see Section 5).

### 4.3 Feature Engineering — Planned and Partially Implemented 🔄

**Name similarity features (9 features):**
- Character-level: Levenshtein ratio, Jaro-Winkler, partial ratio, token-sort ratio, token-set ratio, no-space ratio
- Token-level: Jaccard similarity, TF-IDF cosine (country-specific IDF), rare-token shared IDF score

**Address similarity features (9 features):**
- Same character-level and token-level features as names
- Numeric overlap: exact PIN code match (5–6 digit regex), numeric Jaccard over 2–6 digit tokens

**BM25 retrieval meta-features (5 features):**
- Raw BM25 score, zero-based rank, relative BM25 score (this / best for this probe), BM25 gap, candidate count

**Cross-signal / edge-case features (5+ features):**
- `is_cross_script` flag, `addr_empty` flags, `name_addr_discordance` (distractor detector), `probe_src`

### 4.4 Data Preprocessing Decisions Made ✅

- **Text normalisation**: NFKC normalisation → lowercase → punctuation removal → abbreviation expansion
- **Mined abbreviations from data**: `Pvt→Private`, `Rd→Road`, `Ltd→Limited`, `St→Street`, `Ave→Avenue` (learned from training pairs, not hard-coded)
- **Mined transliterations**: 1,314 Devanagari→Latin token substitutions learned from matched training pairs (e.g., "unique" recovered from "युनिक")
- **French accent handling**: Strip diacritics before script detection (so `é/à/ç` are not misclassified as non-Latin/Hindi)
- **Legal suffix removal**: Inc, Corp, Ltd, LLC, Pvt, SARL, SAS, EURL (French), GmbH (German) — stripped before comparison
- **Vocabulary**: Built on full train+test corpus (all 6 files) so French words have correct IDF — not just training data
- **Validation split**: Fixed hash-based bucket split by S1 entity ID — all candidate pairs for a given S1 entity stay together

---

## 5. Our Proposed Pipeline (What We Are Building)

```
Raw TSVs (train + test)
        │
        ▼
[Stage 0] Text Normalisation
  - Lowercase, NFKC, accent stripping
  - Abbreviation expansion (mined from training pairs)
  - Legal suffix removal
  - Transliteration: Indic script → Latin (mined from labelled pairs)
        │
        ▼
[Stage 1] Blocking / Candidate Generation
  - BM25 index on Source 1, partitioned by country
  - Query: each S2/S3 record retrieves top-10 S1 candidates
  - Relative-score pruning: keep only candidates with score ≥ 80% of best score
  - Output: ~17M candidate pairs (from 17 trillion possible)
        │
        ▼
[Stage 2] Feature Engineering
  - 38+ handcrafted similarity features per candidate pair
  - Computed using rapidfuzz, scikit-learn TF-IDF, regex
        │
        ▼
[Stage 3] LightGBM Pair Classifier (Stage 1)
  - 38-feature binary classifier: "is this pair a true match?"
  - Trained on labelled pairs with negative subsampling
  - Isotonic calibration to convert raw scores → probabilities
        │
        ▼
[Stage 4] Listwise LightGBM (Stage 2 — context-aware)
  - 19 additional features: rank within entity, gap to next candidate,
    entity-level probability statistics (sum, max, share)
  - Refines scores using entity-level context
        │
        ▼
[Stage 5] Threshold / Prefix Selection
  - For each S1 entity, rank candidates by probability
  - Select prefix length k that maximises expected F₀.₅
  - (Lewis 1995 expected-F method)
        │
        ▼
[Stage 6] Output
  - matching_results.tsv (leaderboard-scored)
  - candidate_pairs.tsv (pipeline audit)
  - Validated with utils/validate_submission.py
```

---

## 6. Technical Depth — We Understand the Core Components

### 6.1 Why BM25 for Blocking?

BM25 is an information retrieval ranking function. For a query (S2/S3 probe) and a candidate document (S1 entity):

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot |d| / \text{avgdl})}$$

With $k_1 = 1.2$, $b = 0.75$ (Okapi BM25). We index S1 records and query with S2/S3. This is asymmetric by design — S1 is the reference (deduplicated), so it makes sense to search for a probe's parent in S1.

**Separate field indexing**: Name and address are indexed as separate text fields. A record with a blank address field still gets a fair name-based score (instead of being penalised for missing tokens in a combined field).

### 6.2 Why LightGBM as the Classifier?

LightGBM (Light Gradient Boosting Machine) is:
- Fast to train on millions of pairs (leaf-wise tree growth, histogram-based splitting)
- Handles mixed feature scales (ratios 0–1, BM25 scores 0–20, binary flags) without normalisation
- Produces interpretable feature importances
- Does not require GPU (though GPU accelerates training ~15×)

Training objective: **binary cross-entropy** (log-loss). We train on balanced samples of positive (true match) and negative (non-match) pairs sampled from candidate pairs.

From analysis: the most predictive features are retrieval-based (rank and relative BM25 score dominate), not text similarity — the retrieval stage already does heavy lifting, and LightGBM fine-tunes from there.

### 6.3 Why Is the Metric Non-Trivial?

F₀.₅ macro-averaged means:
- Each S1 entity contributes equally to the final score
- A singleton (no true matches) correctly predicted as empty → **+1.0**
- A singleton with even one wrong match predicted → **+0.0** (full loss on that entity)
- **5.58% of entities are singletons** → these are "free points" if handled correctly

This makes **threshold tuning critically important** — we tune the probability cutoff to maximise this exact metric on a held-out validation split, not accuracy or AUC.

The Lewis 1995 method provides an elegant solution: for each entity, compute the *expected* F₀.₅ at each prefix length k (0 matches, 1 match, 2 matches, ...) and select the k that maximises it. This is a per-entity decision rule, not a single global threshold.

### 6.4 The Out-of-Distribution Challenge (France)

France = ~15% of the test set but has **zero labelled training data**. Simple decomposition:

```
0.85 × F₀.₅(India+US)  +  0.15 × F₀.₅(France)  = Overall Score
```

If India/US performance is ~0.97 but France performs poorly (~0.64) due to OOD, the overall score is ~**0.92**. This gap is the main challenge.

**Root causes we identified:**
1. BM25 and TF-IDF IDF weights built on India+US only → French words have zero document frequency → garbage scores
2. French accented characters (`é, à, ç`) incorrectly flagged as non-Latin (Devanagari/Hindi) by naive ASCII regex
3. French legal suffixes (SARL, SAS, EURL) not stripped — pollute name token sets
4. Isotonic calibration fitted on India/US data → probability estimates wrong for French pairs

**Our mitigations:**
1. Rebuild vocabulary on all 6 files (train + test) — French words get correct IDF
2. Strip diacritics before script detection (`Société` → `Societe` → ASCII only → correctly classified as Latin)
3. Add French legal suffixes to the removal list
4. Use language-agnostic features: postal code exact match, numeric Jaccard (numbers are the same in all languages)
5. Consider multilingual transformer encoder (mDeBERTa-v3-base, MIT license, 278M params) as an additional signal

---

## 7. Preliminary Numbers from Our EDA

| Metric | Value | How We Got It |
|---|---|---|
| Total records (all 6 files) | 24,229,173 | Direct count from all TSVs |
| Labelled true pairs (training) | 7,638,365 | Ground truth TSV |
| Singleton rate (S1 with 0 matches) | 5.58% | Counted from ground truth |
| Blank addresses (probe files) | 610,389 | Data audit |
| Candidate recall @10 (name BM25) | ~97.8% | Sampled blocking test |
| Recall improvement (separate fields) | 90.0% → 95.3% (address-missing subset) | EDA experiment |
| Blocking reduction ratio | ~1,000,000× | 17T → 17M |
| Cross-script pairs in true matches | ~18.7% (sampled) | Sampled EDA |
| Blocking miss attribution: missing addr | ~44% | Blocking miss audit |
| Blocking miss attribution: cross-script | ~37% | Blocking miss audit |

We have not yet trained LightGBM — our current phase is completing feature engineering and training setup.

---

## 8. What We Figured Out Ourselves (Not from GenAI)

The following insights come from our actual hands-on analysis:

- **The singleton rate (5.58%)** — we computed this from the training ground truth TSV, not assumed
- **The OOD decomposition** — we derived that France must score ~0.64 to produce the observed gap between local CV and leaderboard score
- **The blocking miss audit breakdown** — we traced 44% of misses to missing addresses and 37% to cross-script names by inspecting actual failure cases
- **The 0.01% joint failure rate** — only 0.01% of true pairs have zero overlap in both name AND address simultaneously; this tells us we always have at least one signal to work with
- **The Telugu/Andhra Pradesh filter test** — we tested "Telugu script → must be in Andhra Pradesh" as a geographic hard filter, and found it would have silently deleted 874 valid Telangana true pairs — so we rejected it
- **The French accent misclassification** — we discovered that `é/à/ç` were being flagged as non-Latin by the naive `[^\x00-\x7F]` regex, causing French business names to be routed through the cross-script (Devanagari) logic

---

## 9. References

1. Raschka, S. & Mirjalili, V. — *Python Machine Learning* (Packt, 2019) — Feature engineering, gradient boosting
2. Géron, A. — *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (O'Reilly, 2022) — Ensemble methods, model selection
3. Lewis, D.D. (1995) — "Evaluating and Optimizing Autonomous Text Classification Systems" — F-score prefix selection method used in our threshold stage
4. Robertson, S. & Jones, K.S. — "BM25 Okapi Retrieval Function" — Blocking/retrieval stage
5. Amazon ML Challenge 2026 — Problem Statement and Rules PDF — Task specification and metric definition

---

*Document prepared for 25-mark pre-submission evaluation, CSE2032 Machine Learning, BML Munjal University, September 2026.*
