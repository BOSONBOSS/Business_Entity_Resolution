# Comparative Machine Learning Approaches for Business Entity Resolution
## Across Heterogeneous, Noisy, Multi-Script Business Registries

**Course:** CSE2032 — Machine Learning  
**Institution:** BML Munjal University, Dept. of Computer Science & Engineering  
**Batch:** B.Tech 2024, 5th Semester | **Academic Year:** 2026–27  

**Team:**
| Name | Enrollment No. |
|---|---|
| Ananshi Sharma | 240377 |
| Aditya Nadar | 240368 |
| Sumit Singh | 240344 |
| Dhruv Sharma | 240444 |

---

## 1. Introduction & Motivation

Across modern economies, the same real-world business entity exists as fragmented, inconsistent records scattered across dozens of independent databases. A bank's KYC system, a food delivery platform's vendor list, a government tax registry, and a local business directory may all contain records for "Ram Marketing Private Limited" — but they will never use the same ID, the same address format, or even the same script to write the name. No shared primary key links them.

This problem — determining which records across independent sources refer to the same real-world entity — is known in Computer Science and industry as **Business Entity Resolution** (also called **Record Linkage**, **Entity Matching**, or **De-duplication**). It is a foundational data quality and knowledge graph construction problem with high-stakes applications in:

- **Banking & Financial Compliance:** KYC regulations mandate that all records belonging to the same legal entity be consolidated before credit assessment.
- **E-Commerce & Logistics:** Identifying whether "Star Traders" on Platform A is the same business as "Star Traders Pvt. Ltd." on Platform B prevents duplicate vendor onboarding.
- **Government & Taxation:** Tax authorities must deduplicate business identities across registries (GST database vs. MCA21 Corporate Database vs. MSME Udyam Registry).
- **Healthcare & Insurance:** Provider and pharmaceutical supplier databases are frequently redundant. False merges can result in incorrect drug interaction data.

Despite its enormous practical importance, Entity Resolution remains a fundamentally **hard ML problem** due to the inherent noisiness, incompleteness, and heterogeneity of real-world business records. This project undertakes a rigorous experimental investigation into how classical and ensemble machine learning algorithms — as taught in CSE2032 — can be systematically applied and comparatively evaluated to solve Business Entity Resolution.

---

## 2. Problem Statement

### 2.1 Formal Task Definition

Given three independent databases of business records — a **clean, deduplicated reference database (Source 1)** and two **noisy, unlinked probe databases (Source 2 and Source 3)** — the task is to build a supervised binary classification system that determines, for every candidate pair of records $(r_1, r_2)$ where $r_1 \in \text{Source 1}$ and $r_2 \in \text{Source 2} \cup \text{Source 3}$:

$$f(r_1, r_2) \rightarrow y \in \{0, 1\}$$

Where:
- $y = 1$ indicates that $r_1$ and $r_2$ represent the **same real-world business entity**
- $y = 0$ indicates that $r_1$ and $r_2$ represent **distinct, different business entities**

The system must do this by extracting a vector of interpretable, language-agnostic **similarity and structural features** $\mathbf{x} = \phi(r_1, r_2)$ from each pair, training multiple ML classifiers on those features, and producing a systematic **comparative analysis** of model performance.

### 2.2 Input Data Schema

Each business record consists of exactly **four fields**:

| Field | Description | Noise Characteristics |
|---|---|---|
| `entity_id` | Unique record identifier with source prefix (`S1-`, `S2-`, `S3-`) | None — this is the primary key |
| `business_name` | Name of the business as registered in each source | Abbreviations, legal suffix variants, transliterations, typos, website domain names, non-Latin Indic scripts (Devanagari, Tamil, Telugu, Bengali, etc.) |
| `business_address` | Physical business address | Partial addresses, component reordering, landmark references ("Near SBI ATM"), missing PIN codes, fake null values ("null", "n/a", "-"), mixed Hindi-English addresses |
| `country` | Country of registration | `US`, `India` |

**Ground Truth (Training Only):** The ground truth file maps each Source 1 `entity_id` to a comma-separated list of matching `entity_id`s from Sources 2 and 3. An empty list signifies that the Source 1 entity is a **singleton** — a business that exists only in Source 1 with no corresponding record in Source 2 or Source 3.

### 2.3 The Classification Reformulation

We reformulate the multi-database retrieval problem as a **supervised binary pairwise classification problem**:

1. **Supervised Learning Applicability:** The ground truth provides labeled matching pairs $(r_1, r_2, y=1)$. We additionally construct hard non-matching pairs $(r_1, r_2, y=0)$ using three tiers of controlled negative sampling (same-name/different-address, same-city/different-business, random in-country).
2. **Feature Space:** Instead of treating records as opaque strings, we engineer a 15-dimensional interpretable feature vector $\mathbf{x} \in \mathbb{R}^{15}$ encoding string similarity, numeric overlap, and data quality signals.
3. **Comparative Modeling:** With a fixed feature representation, we rigorously compare the decision boundaries, overfitting behavior, and precision-recall tradeoffs of fundamentally different ML algorithm families from the CSE2032 syllabus.

---

## 3. Dataset Description

### 3.1 Raw Data Scale

| File | Records | File Size | Description |
|---|---|---|---|
| `train_source1.tsv` | 2,206,821 | 200 MB | Clean, deduplicated reference businesses (Source 1) |
| `train_source2.tsv` | 5,034,616 | 467 MB | Noisy, unlinked probe businesses (Source 2) |
| `train_source3.tsv` | ~4,500,000 | 480 MB | Noisy, unlinked probe businesses (Source 3) |
| `train_ground_truth.tsv` | 2,206,821 rows | 121 MB | True matching links: S1 entity → {S2/S3 IDs} |
| **Total Training** | **~11.9M records** | **~1.3 GB** | — |

### 3.2 Benchmark Dataset Construction

We construct a **50,000-pair curated benchmark dataset** that is statistically representative, computationally feasible, and scientifically rigorous:

**Positive Pairs ($y = 1$) — 25,000 pairs:**
Stratified sample from ground truth matching links:
- 50% US-based, 50% India-based
- Includes challenging sub-types: script-mismatch (Hindi ↔ English), domain-name records, missing-address records, and abbreviated names, all in proportion to their real-world frequency

**Negative Pairs ($y = 0$) — 25,000 pairs** constructed in three tiers:

| Tier | Count | Description | Challenge |
|---|---|---|---|
| Same-Name, Different-Address | 7,500 | Same normalized name, different city/PIN | A name-only model incorrectly merges these |
| Same-City, Different-Business | 7,500 | Different businesses in the same locality | Partial address overlap misleads address-only models |
| Random In-Country | 10,000 | Randomly sampled non-matching pairs within same country | Baseline distribution coverage |

**Final Benchmark:** 50,000 labeled pairs, 50/50 balanced, ~15 MB, fixed seed (42) for reproducibility.

### 3.3 Key Data Quality Characteristics Found Through EDA

| Challenge | Statistic | Impact on ML Design |
|---|---|---|
| Singleton S1 entities (zero matches) | ~5.6% of S1 | Model must learn "no match" confidently, not always match |
| Non-Latin script names in S2 (Devanagari, Tamil, Telugu, etc.) | ~10–12% of S2 | Standard string similarity = 0.0; need explicit script-mismatch flag |
| Website domain names used as business names in S2/S3 | ~4% of S2/S3 | Word-token Jaccard = 0.0; need character n-gram features |
| Blank or fake-null addresses ("null", "n/a", "-") | ~3.5% of S2/S3 | "null" = "null" produces false 100% address match; must clean |
| Ambiguous names shared by multiple distinct S1 entities | ~48% of S1 | Name similarity alone is insufficient; address is mandatory tiebreaker |
| Legal suffix variants (Pvt/Private, Ltd/Limited, Corp/Corporation) | Pervasive | Must strip before similarity computation |

---

## 4. Scope of the Project

### 4.1 In-Scope: What We Will Do

#### 4.1.1 Text Preprocessing Pipeline (7 Steps)

Applied to both records in every pair before any feature computation:
1. **Lowercasing** — Normalize case uniformity.
2. **Fake-Null Placeholder Removal** — Convert "null", "n/a", "na", "none", "nil", "-" to empty strings.
3. **Unicode Accent Stripping** — NFKD normalization: é→e, à→a, ç→c (critical for European names and prevents accented text from false non-Latin flags).
4. **Legal Suffix Stripping** — Remove corporate form identifiers: Private, Limited, Pvt, Ltd, LLC, Inc, Corp, SARL, SAS, LLP.
5. **Punctuation & Symbol Normalization** — Replace `&` → `and`, remove hyphens, collapse whitespace.
6. **Non-Latin Script Detection** — Flag records with non-Latin Unicode characters (Devanagari: U+0900–U+097F, Tamil: U+0B80–U+0BFF, etc.).
7. **Address Digit Tokenization & Domain Extraction** — Extract numerical components (door/PIN numbers) and domain stems (`.com`, `.in`).

#### 4.1.2 Feature Engineering — 15 Features across 4 Categories

- **Category A: Token-Based Lexical Similarity (5 features):** Name Token Jaccard ($f_1$), Address Token Jaccard ($f_2$), Name Dice Coefficient ($f_3$), Address Numeric Overlap ($f_4$), Name Character Trigram Jaccard ($f_5$).
- **Category B: Edit-Distance Fuzzy Similarity (4 features):** Name Levenshtein Ratio ($f_6$), Name Jaro-Winkler ($f_7$), Address Levenshtein Ratio ($f_8$), Name Token-Sort Ratio ($f_9$).
- **Category C: TF-IDF / Information-Theoretic Similarity (2 features):** Name TF-IDF Cosine ($f_{10}$), Address TF-IDF Cosine ($f_{11}$).
- **Category D: Structural / Data Quality Flags (4 features):** Script Mismatch Flag ($f_{12}$), Address Empty Flag ($f_{13}$), Domain Name Flag ($f_{14}$), Country Match Flag ($f_{15}$).

#### 4.1.3 Comparative ML Model Study — 8 Classifiers

1. **Logistic Regression (L2 / Ridge):** Interpretable weights, probabilistic baseline.
2. **Logistic Regression (L1 / Lasso):** Automatic feature selection via L1 sparsity.
3. **K-Nearest Neighbors ($k$-NN):** Non-parametric instance-based comparison across $k \in \{3, 5, 11\}$ and distance metrics.
4. **Support Vector Machine (Linear Kernel):** Maximum-margin linear hyperplane.
5. **Support Vector Machine (RBF Kernel):** Non-linear kernel boundary (trained on stratified 10k subset).
6. **Decision Tree (CART):** Interpretable white-box rules; comparing unpruned vs. depth-pruned trees.
7. **Random Forest:** Ensemble bagging for variance reduction (100–300 trees).
8. **Gradient Boosting:** Sequential ensemble boosting for bias reduction.

### 4.2 Out-of-Scope

| Item | Reason Excluded |
|---|---|
| Full 11.9M record pairwise search | Beyond course computational scope; 50k benchmark is statistically sufficient |
| Transformer / BERT cross-encoders | Deep Learning is beyond CSE2032 syllabus; requires high-end GPU cluster |
| Industrial BM25 / Lucene blocking index | Systems engineering, not core ML; solved academically via 3-tier negative sampling |
| External API calls (geocoding, scraping) | Data integrity constraint; all features derived strictly from provided records |
| Test dataset inference (no ground truth) | Academic evaluation requires ground truth verification |

---

## 5. End-to-End Methodological Flow (10-Step Lifecycle)

The project executes through a rigorous, sequenced 10-step lifecycle designed to satisfy all CSE2032 Course Outcomes (CO1, CO2, CO3):

```
┌────────────────────────────────────────────────────────────────────────┐
│                   END-TO-END PROJECT FLOW PIPELINE                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Benchmark Dataset Construction                                │
│ • Explode Ground Truth matches (25,000 Positives: 50% US / 50% India)   │
│ • Generate 3-Tier Hard Negatives (25,000 Negatives)                    │
│ • Export clean, balanced benchmark_pairs.csv (50,000 rows, ~15 MB)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Exploratory Data Analysis (EDA)                               │
│ • Text length distributions, missing rate, fake nulls, Indic scripts   │
│ • Quantify name ambiguity (~48% overlap) & generate 4 report plots     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Deterministic Preprocessing Pipeline                          │
│ • 7-step sequence: lowercasing, null scrubbing, accent stripping,      │
│   legal suffix stripping, punctuation, script flags, digit extraction   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Feature Extraction (15 Numerical & Flag Dimensions)           │
│ • Token Jaccard/Dice, Levenshtein, Jaro-Winkler, TF-IDF cosine, flags  │
│ • Export feature_matrix.csv (50,000 × 16)                              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
┌───────────────────────────────────┐ ┌───────────────────────────────────┐
│ Phase 5: Unsupervised Learning    │ │ Phase 6: Supervised Benchmark     │
│ • PCA Dimensionality Reduction    │ │ • 8 Course Models across 5-Fold   │
│   (15D → 2D Projection)           │ │   Stratified Cross-Validation     │
│ • Explained variance ratio & 2D   │ │ • Macro F₀.₅, Precision, Recall,  │
│   decision separability plot      │ │   F₁, AUC-ROC, training latency   │
└───────────────────────────────────┘ └─────────────────┬─────────────────┘
                                                        │
                                                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 7: Hyperparameter Tuning & Bias-Variance Tradeoff Analysis       │
│ • GridSearchCV on top models (Tree depth, Forest trees, GBM rate, C)   │
│ • Validation curves (Underfitting ↔ Overfitting boundary)              │
│ • Learning curves (Sample efficiency from 10% to 100% data)            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 8: Ensemble Modeling & Decision Threshold Optimization           │
│ • Soft Voting Classifier (Probabilistic consensus of top 3 models)    │
│ • Stacking Classifier (Meta-learner on out-of-fold base predictions)   │
│ • Asymmetric threshold sweep (τ ∈ [0.10, 0.90]) maximizing Macro F₀.₅  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 9: Model Interpretability & Error Taxonomy                       │
│ • Feature importance comparison: Logistic regression β vs. RF Gini     │
│ • Confusion matrix decomposition & error taxonomy (hard FP vs. hard FN)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 10: Interactive Demonstration & Academic Report                  │
│ • Interactive live pair evaluation interface                           │
│ • 25–30 page final academic report following BMU university template   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Google Colab & Cloud Execution Architecture

To ensure 100% reproducibility, seamless team collaboration, and zero environment conflicts, the entire experimental suite is architected to run in **Google Colab** using free-tier resources without out-of-memory crashes.

### 6.1 Directory Structure on Google Drive

```
/MyDrive/Business_Entity_Resolution/
│
├── data/
│   ├── raw/                 # train_source1.tsv, train_source2.tsv, train_ground_truth.tsv
│   └── processed/           # benchmark_pairs.csv (15 MB), feature_matrix.csv (7 MB)
│
├── results/
│   ├── plots/               # EDA charts, PCA 2D plot, ROC curves, PR curves, Confusion matrices
│   └── tables/              # model_comparison.csv, feature_importances.csv, gridsearch_results.csv
│
└── notebooks/
    ├── 01_Data_Prep_and_EDA.ipynb
    ├── 02_Feature_Engineering_and_PCA.ipynb
    └── 03_Model_Training_and_Evaluation.ipynb
```

### 6.2 Three-Notebook Modular Breakdown

| Notebook | Purpose | Input | Output | Memory Footprint |
|---|---|---|---|---|
| **`01_Data_Prep_and_EDA.ipynb`** | Chunked extraction of 50K benchmark, negative sampling, EDA visualisations | `data/raw/*.tsv` | `benchmark_pairs.csv`, 4 EDA plots | Chunked reading (`chunksize=100K`) prevents RAM spike |
| **`02_Feature_Engineering_and_PCA.ipynb`** | 7-step text cleaning, 15-feature extraction, PCA 2D projection | `benchmark_pairs.csv` | `feature_matrix.csv`, PCA plot, Correlation heatmap | ~150 MB RAM; runs in ~2 minutes |
| **`03_Model_Training_and_Evaluation.ipynb`** | 5-Fold CV on 8 models, GridSearchCV, Ensembles, ROC/PR curves, Live Demo | `feature_matrix.csv` | Comparison tables, final figures, Interactive widget | ~300 MB RAM; runs in ~5 minutes |

### 6.3 RAM Optimization & Hardware Scalability Protocol
1. **One-Time Heavy Ingestion:** The 2.4 GB raw files are accessed **only once** in Notebook 1 via chunked streaming.
2. **Lightweight Downstream Execution:** All subsequent modeling, feature extraction, and ensembling operate strictly on the 15 MB `benchmark_pairs.csv` and 7 MB `feature_matrix.csv`.
3. **No GPU Required:** All models use CPU-optimized `scikit-learn` algorithms, running on Colab's standard 2-core CPU runtime in under 10 minutes total execution time.
4. **Live Viva Widget:** Notebook 3 concludes with an interactive `@ipywidgets` interface allowing the faculty evaluation panel to type any two business names and addresses live and view preprocessing transformations, feature values, and the ensemble classification decision in real time.

---

## 7. Evaluation Methodology

### 7.1 Primary Metric: Macro F₀.₅

$$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

**Rationale:** In entity resolution, incorrectly merging two distinct businesses (False Positive) corrupts financial KYC and catalog data. $F_{0.5}$ formally encodes this asymmetric cost by weighting Precision 4× as much as Recall in the harmonic mean.

### 7.2 Supporting Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| Precision | $TP / (TP + FP)$ | Of all predicted matches, what fraction are real? |
| Recall | $TP / (TP + FN)$ | Of all real matches, what fraction did we discover? |
| Macro F₁ | Harmonic mean of P and R | Standard balanced baseline metric |
| AUC-ROC | Area under ROC curve | Threshold-independent ranking quality |
| Training Latency | Seconds | Algorithmic computational efficiency |

### 7.3 Evaluation Protocol

- **Stratified 5-Fold Cross-Validation** at the entity level (all pairs containing a given S1 entity remain in the same fold to prevent memorization leakage).
- **Hold-Out Test Split:** A strictly isolated 20% test partition (10,000 pairs) evaluated once at the very end.

---

## 8. Research Questions This Project Answers

| # | Research Question |
|---|---|
| **RQ1** | Can a 15-dimensional interpretable feature vector achieve statistically robust separation between matching and non-matching record pairs? |
| **RQ2** | Which ML algorithm family achieves the highest Macro F₀.₅, and what structural properties explain the advantage? |
| **RQ3** | How does each model's bias-variance profile manifest? Where does Logistic Regression underfit? Where does an unpruned Decision Tree overfit? |
| **RQ4** | Does combining models (Soft Voting or Stacking) outperform the best individual model, and which combinations produce complementary errors? |
| **RQ5** | How much does the script-mismatch flag and address numeric overlap improve classification on Hindi-English record pairs vs. a baseline ignoring script information? |
| **RQ6** | Which features contribute most to classification accuracy, and does this align with EDA domain intuitions (address > name for hard cases)? |
| **RQ7** | What is the optimal decision threshold $\tau^*$ that maximizes Macro F₀.₅, and how much does tuning away from 0.5 improve performance? |

---

## 9. What We Will Discover & Report

### 9.1 Quantitative Findings
- Comprehensive 8-model comparison table across all 5 evaluation metrics.
- Hyperparameter sensitivity curves ($C$ vs. score, tree depth vs. score, $n\_estimators$ vs. score).
- Consolidated feature importance rankings across linear, tree, and ensemble paradigms.
- Empirical threshold optimization curve demonstrating the gain from tuning $\tau^*$.

### 9.2 Qualitative Findings & Academic Contributions
- **Error Taxonomy:** Detailed failure analysis categorizing remaining false positives and false negatives.
- **PCA Visual Representation:** 2D mapping proving that handcrafted string similarity dimensions create clear linear and non-linear cluster boundaries.
- **Interpretable Decision Rules:** Human-readable if-then-else logic extracted from pruned Decision Trees for auditable KYC deployment.
- **Language-Agnostic Feature Efficacy:** Proving that structural features (numeric overlap, script flags) bridge the cross-script gap without heavy multilingual neural networks.

---

## 10. Final Deliverables

| Deliverable | Location | Description |
|---|---|---|
| **Benchmark Dataset** | `data/processed/benchmark_pairs.csv` | 50,000 labeled, stratified business record pairs (~15 MB) |
| **Feature Matrix** | `data/processed/feature_matrix.csv` | 15-feature vectors for all 50,000 pairs (~7 MB) |
| **Preprocessing Module** | `src/preprocess.py` | 7-step text normalization pipeline |
| **Feature Extraction Module** | `src/features.py` | All 15 feature computations |
| **Model & Evaluation Modules** | `src/models.py`, `src/evaluate.py` | Training, 5-Fold CV, metric calculations |
| **Google Colab Notebooks** | `notebooks/01_EDA.ipynb`, `02_Features.ipynb`, `03_Models.ipynb` | 3 self-contained runnable Colab notebooks |
| **Interactive Demo** | `src/app.py` & Colab Widget | Live pair prediction interface for viva demonstration |
| **Final Project Report** | `01_submission_deliverables/` | 25–30 page report following university template |
| **Presentation Deck** | `01_submission_deliverables/` | 12–15 slide presentation deck for evaluation viva |
