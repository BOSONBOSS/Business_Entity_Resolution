# CSE2032 ML Project — Master Implementation Plan
## Business Entity Resolution: A Comparative Machine Learning Study

**Course:** CSE2032 — Machine Learning | **Semester:** 5th | **AY:** 2026-27  
**Team:** Ananshi Sharma (240377) | Aditya Nadar (240368) | Sumit Singh (240344) | Dhruv Sharma (240444)  
**Evaluation Weightage:** 40% (End-Semester Experiential Learning Project)  
**Last Updated:** November 2026

---

> [!IMPORTANT]
> **This plan is fully independent of any competition.** From this point, all references are to the academic ML project only. The dataset is real-world industry data we are using for research purposes.

---

## 🗺️ Table of Contents

1. [The Big Picture — What Are We Doing?](#1-the-big-picture)
2. [Dataset Anatomy — What Do We Have?](#2-dataset-anatomy)
3. [Problem Reformulation — Hackathon → ML Course](#3-problem-reformulation)
4. [Phase 1 — Project Setup & Environment](#4-phase-1-project-setup)
5. [Phase 2 — Data Sampling & Benchmark Construction](#5-phase-2-data-sampling)
6. [Phase 3 — Exploratory Data Analysis (EDA)](#6-phase-3-eda)
7. [Phase 4 — Text Preprocessing Pipeline](#7-phase-4-preprocessing)
8. [Phase 5 — Feature Engineering](#8-phase-5-feature-engineering)
9. [Phase 6 — ML Model Comparison (The Core)](#9-phase-6-models)
10. [Phase 7 — Hyperparameter Tuning & Cross-Validation](#10-phase-7-tuning)
11. [Phase 8 — Ensemble Methods](#11-phase-8-ensembles)
12. [Phase 9 — Model Interpretability & Error Analysis](#12-phase-9-interpretability)
13. [Phase 10 — Final Report & Deliverables](#13-phase-10-report)
14. [Team Responsibility Matrix](#14-team-matrix)
15. [Mapping to Course Outcomes (CO1/CO2/CO3)](#15-course-outcome-mapping)
16. [What Gets Graded — Report Template Checklist](#16-grading-checklist)
17. [Research Gaps & Literature Review](#17-research-gaps)
18. [Final Deliverables Summary](#18-deliverables)

---

## 1. The Big Picture

### What Is The Problem?

Imagine a country has three separate databases of businesses:
- **Source 1 (S1):** A "clean" master registry — think the Ministry of Corporate Affairs. Each entry is unique, deduplicated.
- **Source 2 (S2):** A noisy database, like records scraped from food delivery apps. Same business, but the name might be in Hindi, the address abbreviated, or the entry a website URL.
- **Source 3 (S3):** Another noisy database, perhaps from a local business listing platform.

The question is: **Which record in S2 and S3 refers to the same real-world business as a record in S1?**

This is called **Business Entity Resolution** — also known as **Record Linkage** or **De-duplication** in academia. It is a critically important ML problem in e-commerce, banking (KYC), government databases, and healthcare.

### Why Is This Hard?

The same business can appear as:
| Attribute | Source 1 (Clean) | Source 2 (Noisy) |
|---|---|---|
| Name | "Ram Marketing Private Limited" | "राम मार्केटिंग प्राइवेट लिमिटेड" (Hindi!) |
| Name | "Dhan Investments Limited" | "dhaninvestments.com" (a website!) |
| Name | "Star Services Pvt Ltd" | "Star Services" |
| Address | "1795 Westchester Drive, High Point, NC" | "1795 W Dr, High Pt" |
| Address | "Block 5, Sector 7, New Delhi" | "null" (literally the word null) |

No traditional `JOIN` or exact string match can solve this. This is precisely **why Machine Learning is needed.**

### What Is Our Goal (Course Project)?

We build a supervised ML classification pipeline that:
1. Takes a **pair of business records** $(r_1, r_2)$ as input.
2. Extracts **numerical features** (similarities, flags) from them.
3. **Predicts** whether the pair refers to the **same real-world business** (class 1) or different businesses (class 0).
4. **Compares multiple ML algorithms** from the CSE2032 syllabus to identify which works best and *why*.

---

## 2. Dataset Anatomy

### The Raw Files We Have

| File | Size | Approx. Rows | Contents |
|---|---|---|---|
| `train_source1.tsv` | 200 MB | ~2.2M | Clean reference businesses |
| `train_source2.tsv` | 467 MB | ~5.0M | Noisy probe records |
| `train_source3.tsv` | 480 MB | ~4.5M | Noisy probe records |
| `train_ground_truth.tsv` | 121 MB | ~2.2M | True match links (S1 → S2/S3) |

> [!NOTE]
> We work **exclusively with the training files** because only they have ground truth labels. The test files are discarded from our academic project.

### Schema of Each Record (4 Columns)

```
entity_id       business_name                   business_address               country
---------       -------------                   ----------------               -------
S1-925783039    Orelee's Barbershop             1795 Westchester Drive, ...    US
S2-166376419    राम मार्केटिंग प्राइवेट लिमिटेड  KH NO. -570/13, NEW DELHI    India
```

### The Ground Truth Structure

The `train_ground_truth.tsv` tells us the true matches:
```
source1_entity_id    matched_entity_ids
S1-965667            S2-681193310,S2-743505751,S3-775321672
S1-55344266          S2-249013014,S3-478195123
S1-343815751         (empty — this S1 entity has NO matches, it's a singleton)
```

---

## 3. Problem Reformulation — Hackathon → ML Course

### The Core Transformation

```
ORIGINAL PROBLEM (Industrial Scale)        →    OUR ML COURSE PROJECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"Find ALL matching pairs in 24M records"   →    "CLASSIFY if a given pair matches"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Two massive databases               →    Input: A pair of business records
Output: Full bipartite matching graph      →    Output: 0 (no match) or 1 (match)
Scale: 17.27 trillion pairs               →    Scale: 50,000 curated pairs
Metric: Macro F₀.₅ (competition-specific) →    Metric: Precision, Recall, F₁, AUC-ROC
Infra: Custom BM25 engine + transformers  →    Infra: scikit-learn on a laptop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

This transformation makes it:
- **Feasible:** Fits in RAM (<500 MB), runs in under an hour.
- **Academic:** Requires the exact algorithms in the CSE2032 syllabus.
- **Rigorous:** Allows controlled comparison of models with cross-validation.

---

## 4. Phase 1 — Project Setup & Environment

**Who:** Everyone together on Day 1 | **Time Estimate:** 2 hours

### 4.1 Folder Structure

Create this exact directory structure for the project:

```
ml_project/
│
├── data/
│   ├── raw/                     ← Symlinks or copies of the TSV files
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   │
│   └── processed/               ← Generated files go here
│       ├── benchmark_pairs.csv       ← 50K pairs with labels (Step 5)
│       └── feature_matrix.csv        ← Extracted features (Step 8)
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Comparison.ipynb
│   ├── 05_Hyperparameter_Tuning.ipynb
│   └── 06_Ensemble_Analysis.ipynb
│
├── src/                         ← Clean, importable Python modules
│   ├── preprocess.py
│   ├── features.py
│   ├── models.py
│   ├── evaluate.py
│   └── visualize.py
│
├── results/                     ← All charts, tables, CSVs from experiments
│   ├── plots/
│   └── tables/
│
├── report/                      ← Final project report
│   └── ML_Project_Final_Report.docx
│
├── requirements.txt
└── README.md
```

### 4.2 Python Environment & Libraries

**`requirements.txt`**:
```
pandas==2.1.0
numpy==1.26.0
scikit-learn==1.3.2
matplotlib==3.8.0
seaborn==0.13.0
rapidfuzz==3.5.2        # Fast string similarity: Levenshtein, Jaro-Winkler
nltk==3.8.1             # Tokenization
jupyter==1.0.0
openpyxl==3.1.2         # For Excel exports of result tables
```

**Setup Command:**
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
pip install -r requirements.txt
```

### 4.3 Why These Specific Libraries?

| Library | Purpose | Course Relevance |
|---|---|---|
| `scikit-learn` | All ML models (LogReg, SVM, Tree, RF, GBM, KNN) | CO2: Model development |
| `pandas` / `numpy` | Data loading, manipulation, vectorized operations | CO1: Feature engineering |
| `rapidfuzz` | Levenshtein, Jaro-Winkler similarity at C-level speed | CO1: Feature extraction |
| `matplotlib` / `seaborn` | ROC curves, confusion matrices, feature importance plots | CO3: Evaluation |
| `nltk` | Tokenization for Jaccard similarity | CO1: Text preprocessing |

---

## 5. Phase 2 — Data Sampling & Benchmark Construction

**Who:** Ananshi Sharma (240377) | **Time Estimate:** 4-6 hours

> [!IMPORTANT]
> **This is the most critical step.** A bad sampling strategy gives unreliable results. The professor will ask how we built our training dataset.

### 5.1 Concept: What Is A "Pair"?

The raw dataset gives us records. We need **pairs** to train a classifier. Each pair has:
- `r1`: A business from Source 1
- `r2`: A business from Source 2 or Source 3
- `label`: 1 if they're the same business (from ground truth), 0 if they're different

### 5.2 Positive Pair Sampling (y=1, True Matches)

**Step 1:** Parse `train_ground_truth.tsv` to extract all true matching pairs.
**Step 2:** Join each `source1_entity_id` with its corresponding S1 record (name, address, country).
**Step 3:** Join each matched `S2/S3` ID with its record from `train_source2.tsv` / `train_source3.tsv`.

```python
import pandas as pd

# Load data in chunks to avoid RAM overflow
s1 = pd.read_csv("data/raw/train_source1.tsv", sep="\t", dtype=str)
s2 = pd.read_csv("data/raw/train_source2.tsv", sep="\t", dtype=str)
gt = pd.read_csv("data/raw/train_ground_truth.tsv", sep="\t", dtype=str)

# Explode ground truth: one row per positive pair
gt_exploded = gt.assign(
    matched_entity_ids=gt['matched_entity_ids'].str.split(',')
).explode('matched_entity_ids')
gt_exploded = gt_exploded[gt_exploded['matched_entity_ids'].notna()]

# Join: S1 record + S2/S3 record
positive_pairs = gt_exploded.merge(
    s1, left_on='source1_entity_id', right_on='entity_id', suffixes=('', '_s1')
).merge(
    s2, left_on='matched_entity_ids', right_on='entity_id', suffixes=('_s1', '_s2')
)

positive_pairs['label'] = 1
```

**How many positives?** The ground truth has ~7.6M total matching links. We will **stratified sample** 25,000 from these.

**Key stratification criteria:**
- Include easy positives (name matches perfectly).
- Include hard positives (Hindi names, missing addresses, domain-name matches) — these are the challenging but realistic cases.
- Balance by country: US (~50%), India (~50%).

### 5.3 Negative Pair Sampling (y=0, Non-Matches)

**This is where most ML beginners fail.** Simply picking random pairs as negatives creates "easy" negatives that any model can distinguish trivially. This inflates accuracy and gives a false sense of model performance.

We create **three types of hard negatives:**

#### Type A: Same-Name, Different-Address Negatives (~30% of negatives)
Businesses that share the same name but are in different locations.

**Why hard?** A naive name-matching system would merge these.
**How to find them?** Find S1 records with identical normalized names, then pair them with S2 records that match only on name but not address.

```python
# Find S1 entities that share the same normalized name
# (normalized = lowercase, strip punctuation, remove legal suffixes)
s1['normalized_name'] = s1['business_name'].str.lower().str.replace(r'[^\w\s]', '')
name_groups = s1.groupby('normalized_name').filter(lambda x: len(x) > 1)
# Pick pairs from different groups as negatives
```

#### Type B: Same Country, Same City, Different Business (~30% of negatives)
Two unrelated businesses in the same city — their addresses will partially match (same city/state), but they are different companies.

**Why hard?** An address-heavy model might merge these.

#### Type C: Random In-Country Negatives (~40% of negatives)
Randomly paired non-matching records within the same country.

**Why in-country?** We observed 100% of real matches stay within country. So cross-country pairs are trivially "no match" — including them would inflate accuracy unfairly.

### 5.4 Final Benchmark Dataset

```
benchmark_pairs.csv (50,000 rows)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Columns:
  s1_entity_id, s1_name, s1_address, s1_country
  s2_entity_id, s2_name, s2_address, s2_country
  label          (0 or 1)
  pair_difficulty (easy_positive, hard_positive, easy_negative, hard_negative_name, hard_negative_addr)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Distribution:
  Positives (label=1): 25,000 (50%)
  Negatives (label=0): 25,000 (50%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> [!TIP]
> **Why 50/50 balance?** Real-world entity resolution is extremely imbalanced (~1 match per 1,000 candidates). We balance it 50/50 to let models learn effectively during training, then test on a more realistic 1:4 or 1:9 imbalanced test set to measure real-world behavior.

**Save with reproducibility seed:**
```python
RANDOM_SEED = 42  # Fixed for reproducibility
positives = positive_df.sample(25000, random_state=RANDOM_SEED)
negatives = negative_df.sample(25000, random_state=RANDOM_SEED)
benchmark = pd.concat([positives, negatives]).sample(frac=1, random_state=RANDOM_SEED)
benchmark.to_csv("data/processed/benchmark_pairs.csv", index=False)
```

---

## 6. Phase 3 — Exploratory Data Analysis (EDA)

**Who:** Ananshi Sharma (240377) | **Time Estimate:** 6-8 hours  
**Deliverable:** `notebooks/01_EDA.ipynb` with all charts + a written summary section in the report

> [!IMPORTANT]
> EDA is **not** just making pretty graphs. EDA must **answer specific questions** that **justify every design choice** you make later. The professor will ask: "Why did you use this feature?" — the answer must come from EDA.

### 6.1 Dataset-Level EDA (What Does The Data Look Like?)

**Question 1: What is the country distribution?**
```python
print(s1['country'].value_counts())
# Expected: US: ~1.1M, India: ~1.0M
```
*Design Impact:* Confirms we must handle English and Indic scripts.

**Question 2: What fraction of S1 entities have zero matches (singletons)?**
```python
no_match = gt[gt['matched_entity_ids'].isna() | (gt['matched_entity_ids'] == '')]
singleton_rate = len(no_match) / len(gt)
print(f"Singleton rate: {singleton_rate:.1%}")
# Expected: ~5-6%
```
*Design Impact:* About 5% of S1 records are genuinely unmatchable. Our classifier must learn to predict "no match" for these, not just always predict a match.

**Question 3: How many addresses are blank/fake-null?**
```python
placeholder_strings = {'null', 'n/a', 'na', 'nan', 'nil', 'none', '-', ''}
s2['addr_blank'] = s2['business_address'].str.lower().str.strip().isin(placeholder_strings)
print(f"S2 blank address rate: {s2['addr_blank'].mean():.1%}")
# Expected: ~3.5% blank or fake-null
```
*Design Impact:* We must clean these before computing address similarity, or "null" = "null" looks like a perfect address match!

**Question 4: What fraction of S2/S3 names are non-Latin (Hindi, Tamil, etc.)?**
```python
import unicodedata
def is_non_latin(s):
    if not isinstance(s, str): return False
    return any(unicodedata.category(c) == 'Lo' and ord(c) > 0x024F for c in s)

s2['is_non_latin'] = s2['business_name'].apply(is_non_latin)
print(f"S2 non-Latin name rate: {s2['is_non_latin'].mean():.1%}")
# Expected: ~10-12%
```
*Design Impact:* For these pairs, name similarity will be artificially 0. We need a feature flag for this, and must weight address features higher.

**Question 5: What fraction of names look like website domains?**
```python
s2['is_domain'] = s2['business_name'].str.contains(r'\.(com|org|net|in|co)$', regex=True, case=False)
print(f"S2 domain name rate: {s2['is_domain'].mean():.1%}")
# Expected: ~4%
```
*Design Impact:* Word-token Jaccard will give 0% similarity for `dhaninvestments.com` vs `Dhan Investments`. We need character n-gram features.

### 6.2 Pair-Level EDA (What Makes a True Match Different From a Non-Match?)

This is where we look at the **feature distributions before engineering features** — it tells us which raw signals are most predictive.

**Question 6: On true matching pairs, what is the token Jaccard distribution?**
```python
# On our positive_pairs sample
from rapidfuzz.distance import Jaccard
positive_pairs['name_jaccard'] = positive_pairs.apply(
    lambda r: Jaccard.normalized_similarity(
        r['business_name_s1'].lower().split(),
        r['business_name_s2'].lower().split()
    ), axis=1
)
positive_pairs['name_jaccard'].hist(bins=50)
```

Expected insight: **Bimodal distribution** — many positives have Jaccard=1.0 (easy), but a significant tail around Jaccard=0.0 (hard — script mismatch or domain names). This bimodal pattern proves we need multiple features, not just name similarity.

**Question 7: On true pairs, name Jaccard vs. address Jaccard — which is more reliable?**

Create a scatter plot: `name_jaccard` (x-axis) vs `address_jaccard` (y-axis), colored by `is_non_latin`.

Expected insight: When `name_jaccard ≈ 0` (non-Latin script pairs), `address_jaccard` is often still > 0.5 (addresses are usually written in English/digits). This proves address is the **fallback signal for script-mismatch pairs**.

**Question 8: Distribution of hard negative vs. positive similarities**
```python
# Plot name_jaccard for positives vs hard_negative_name pairs
import seaborn as sns
sns.kdeplot(data=benchmark_pairs, x='name_jaccard', hue='label')
```

Expected insight: Hard negatives overlap with positives in name similarity — this proves name similarity alone is insufficient and justifies multi-feature engineering.

### 6.3 EDA Output Summary Table

Your EDA section in the report must include this exact table with real numbers filled in:

| Finding | Statistic | Design Decision It Drives |
|---|---|---|
| Country distribution | US: X%, India: Y% | Must handle English & Devanagari scripts |
| S1 singleton rate | ~5.6% | "No match" class must be learnable |
| Blank/fake-null addresses | ~3.5% of S2 | Step 1 of preprocessing: placeholder cleaning |
| Non-Latin name records | ~10% of S2 | Add `is_script_mismatch` feature flag |
| Domain name records | ~4% of S2 | Add character n-gram similarity feature |
| Name ambiguity | ~48% of S1 share names | Address similarity must be a primary feature |
| True pairs with name_jaccard=0 | ~15% of positives | Name similarity alone → biased model |

---

## 7. Phase 4 — Text Preprocessing Pipeline

**Who:** Ananshi Sharma (240377) | **Time Estimate:** 4-6 hours  
**Deliverable:** `src/preprocess.py` — a clean, importable Python module

### 7.1 Why Preprocessing Matters

Before computing any similarity between records, we must **standardize** the text. Consider:
- S1: `"Dhan Investments Private Limited"`
- S2: `"dhan investments pvt. ltd."`

Without preprocessing: Jaccard = 2/8 = 0.25 (only "dhan", "investments" match)  
After preprocessing: Jaccard = 2/3 = 0.67 (after removing "Private/pvt", "Limited/ltd")

Preprocessing directly increases the **signal-to-noise ratio** for our features.

### 7.2 The 7-Step Preprocessing Pipeline

Each step is a function in `src/preprocess.py`:

#### Step 0: Lowercase
```python
def to_lower(text: str) -> str:
    return text.lower() if isinstance(text, str) else ""
```
Trivial but essential — "STAR" and "star" should be identical.

#### Step 1: Handle Fake-Null Placeholders
```python
FAKE_NULLS = {'null', 'n/a', 'na', 'nan', 'nil', 'none', '-', 'n.a.', 'n.a'}

def clean_placeholder(text: str) -> str:
    """Convert fake null strings to empty string."""
    t = text.strip().lower() if isinstance(text, str) else ""
    return "" if t in FAKE_NULLS else text
```

**Why critical?** If we don't do this, two completely different businesses that both have address="null" will have an address Jaccard similarity of 1.0 — a false match signal!

#### Step 2: Strip Accents (For French/European Text)
```python
import unicodedata

def strip_accents(text: str) -> str:
    """Convert é→e, à→a, ç→c etc. for uniform comparison."""
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if unicodedata.category(c) != 'Mn'  # Mn = Non-spacing Mark (accent)
    )
```

**Why critical?** "Société Générale" → "Societe Generale". Without this, the accent characters would cause zero overlap with the English version.

#### Step 3: Remove Legal Suffixes
```python
LEGAL_SUFFIXES = {
    # English
    'private', 'limited', 'pvt', 'ltd', 'llc', 'inc', 'corp', 'corporation',
    'co', 'company', 'lp', 'llp', 'plc',
    # Indian
    'pvtltd', 'private limited',
    # French
    'sarl', 'sas', 'sa', 'eurl',
}

def remove_legal_suffixes(name: str) -> str:
    """Strip legal form identifiers from business names."""
    tokens = name.split()
    cleaned = [t for t in tokens if t.lower().rstrip('.,') not in LEGAL_SUFFIXES]
    return ' '.join(cleaned)
```

**Why critical?** "Ram Marketing Pvt Ltd" and "Ram Marketing Private Limited" become identical after normalization. This directly doubles the number of exact-match pairs we find.

#### Step 4: Normalize Punctuation & Whitespace
```python
import re

def normalize_punctuation(text: str) -> str:
    """Replace & → and, remove hyphens, collapse whitespace."""
    text = text.replace('&', 'and').replace('-', ' ')
    text = re.sub(r'[^\w\s]', ' ', text)   # Remove all non-alphanumeric
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

**Why critical?** "AT&T" vs "AT and T" vs "A.T.T" — all refer to the same entity.

#### Step 5: Detect Non-Latin Script Flag
```python
def detect_script_flag(text: str) -> bool:
    """Returns True if text contains significant non-Latin characters."""
    if not isinstance(text, str): return False
    non_latin_chars = sum(1 for c in text if ord(c) > 0x024F)  # Beyond Extended Latin
    return non_latin_chars > 2  # Threshold to avoid false positives from accents
```

**Why critical?** This flag becomes a feature. When a pair has `script_mismatch=True`, the model knows not to trust the name similarity score.

#### Step 6: Domain Name Detection & Extraction
```python
def extract_domain_core(text: str) -> str:
    """If name is a domain like 'dhaninvestments.com', extract 'dhaninvestments'."""
    domain_match = re.match(r'^([a-z0-9]+)\.(com|org|net|in|co\.in)$', text.strip().lower())
    if domain_match:
        return domain_match.group(1)  # 'dhaninvestments'
    return text
```

**Why critical?** "dhaninvestments.com" after extraction becomes "dhaninvestments", which now has character n-gram overlap with "Dhan Investments".

#### Step 7: Extract Numeric Tokens (From Addresses)
```python
def extract_address_numbers(address: str) -> set:
    """Extract all numeric tokens from address string."""
    if not isinstance(address, str): return set()
    return set(re.findall(r'\b\d{3,}\b', address))  # Only 3+ digit numbers
```

**Why critical?** Two businesses at "1795 Westchester Drive" and "1796 Westchester Drive" are DIFFERENT. The numeric tokens extract the discriminating building/PIN numbers.

### 7.3 Full Preprocessing Pipeline Function

```python
def preprocess_record(name: str, address: str) -> dict:
    """Apply full preprocessing pipeline to a single business record."""
    # Clean each field
    name_clean = remove_legal_suffixes(
        normalize_punctuation(
            strip_accents(
                to_lower(clean_placeholder(name))
            )
        )
    )
    
    addr_clean = normalize_punctuation(
        strip_accents(
            to_lower(clean_placeholder(address))
        )
    )
    
    # Extract metadata flags
    is_script = detect_script_flag(name)
    is_domain = bool(re.match(r'^[a-z0-9]+\.(com|org|net|in)$', name_clean))
    addr_nums = extract_address_numbers(addr_clean)
    
    return {
        'name_clean': name_clean,
        'addr_clean': addr_clean,
        'is_non_latin': is_script,
        'is_domain': is_domain,
        'addr_numbers': addr_nums,
    }
```

---

## 8. Phase 5 — Feature Engineering

**Who:** Aditya Nadar (240368) | **Time Estimate:** 8-10 hours  
**Deliverable:** `src/features.py` — extracts all features from a record pair

> [!IMPORTANT]
> **Feature engineering is what separates a novice ML project from a real one.** Our models receive a fixed vector of numerical features. The richer and more informative these features are, the better any model can perform. This section directly maps to CO1 (feature engineering).

### 8.1 Feature Taxonomy

We build **15 features** across 4 categories:

```
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE VECTOR (15 features)             │
├────────────────────┬────────────────────────────────────────┤
│ Category A (5)     │ Token-Based Similarity (Jaccard, Dice) │
│ Category B (4)     │ Edit-Distance Similarity (Fuzzy)       │
│ Category C (3)     │ TF-IDF / Information-Theoretic         │
│ Category D (3)     │ Structural / Data Quality Flags        │
└────────────────────┴────────────────────────────────────────┘
```

### 8.2 Category A: Token-Based Similarity Features

These treat names/addresses as **bags of words** and measure how much overlap there is.

#### Feature 1: Name Token Jaccard Similarity
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Where $A$ and $B$ are the **sets of tokens** in the two names.

```python
def token_jaccard(str1: str, str2: str) -> float:
    """Jaccard similarity between token sets."""
    if not str1 and not str2: return 1.0  # Both empty → consider same
    if not str1 or not str2: return 0.0   # One empty → no overlap
    tokens1, tokens2 = set(str1.split()), set(str2.split())
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union) if union else 0.0

# Feature: f1_name_jaccard
row['f1_name_jaccard'] = token_jaccard(r1['name_clean'], r2['name_clean'])
```

**Range:** 0.0 (no token overlap) to 1.0 (identical token sets)  
**Expected discriminating power:** High for English pairs, low for Hindi-English pairs (intentionally — the flag in Category D will handle this).

#### Feature 2: Address Token Jaccard Similarity
Same formula applied to addresses.

```python
# Feature: f2_addr_jaccard
row['f2_addr_jaccard'] = token_jaccard(r1['addr_clean'], r2['addr_clean'])
```

**Expected discriminating power:** Extremely high. Address token overlap is the best single discriminator.

#### Feature 3: Name Dice Coefficient
$$\text{Dice}(A, B) = \frac{2|A \cap B|}{|A| + |B|}$$

Similar to Jaccard but weights shared tokens more heavily when both names are short.

```python
def dice_coefficient(str1: str, str2: str) -> float:
    t1, t2 = set(str1.split()), set(str2.split())
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    return 2 * len(t1 & t2) / (len(t1) + len(t2))

# Feature: f3_name_dice
row['f3_name_dice'] = dice_coefficient(r1['name_clean'], r2['name_clean'])
```

#### Feature 4: Address Numeric Overlap
Checks if the specific numbers in addresses match — ZIP codes, building numbers.

```python
def numeric_overlap(nums1: set, nums2: set) -> float:
    """Jaccard on digit-sequence sets."""
    if not nums1 and not nums2: return 0.5  # Both empty: uncertain
    if not nums1 or not nums2: return 0.0   # One has numbers, other doesn't
    return len(nums1 & nums2) / len(nums1 | nums2)

# Feature: f4_addr_numeric_overlap
row['f4_addr_numeric_overlap'] = numeric_overlap(r1['addr_numbers'], r2['addr_numbers'])
```

**Why this is special:** "123 Main St" vs "456 Main St" have high token Jaccard (0.5) but zero numeric overlap — they are DIFFERENT addresses. This feature catches that.

#### Feature 5: Character Trigram Similarity (For Domain Names)

Instead of splitting into words, we split into overlapping 3-character chunks:  
"dhan" → {"dha", "han"}  
"dhaninvestments" → {"dha", "han", "ani", "nin", "inv", ...}

```python
def char_ngrams(text: str, n: int = 3) -> set:
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def char_ngram_jaccard(str1: str, str2: str, n: int = 3) -> float:
    g1, g2 = char_ngrams(str1, n), char_ngrams(str2, n)
    if not g1 and not g2: return 1.0
    if not g1 or not g2: return 0.0
    return len(g1 & g2) / len(g1 | g2)

# Feature: f5_name_char3gram_jaccard
row['f5_name_char3gram_jaccard'] = char_ngram_jaccard(r1['name_clean'], r2['name_clean'])
```

**Why this matters:** "dhaninvestments" (extracted from domain) has ~60% character trigram overlap with "dhan investments" — even though their word-token Jaccard is 0!

### 8.3 Category B: Edit-Distance / Fuzzy String Features

These measure how many character-level edits are needed to transform one string into another.

#### Feature 6: Name Levenshtein Ratio
The **edit distance** counts minimum insertions, deletions, substitutions to convert one string to another. Normalized to [0, 1]:

$$\text{LevenshteinRatio}(a, b) = 1 - \frac{\text{EditDistance}(a, b)}{\max(|a|, |b|)}$$

```python
from rapidfuzz.distance import Levenshtein

def levenshtein_ratio(str1: str, str2: str) -> float:
    if not str1 and not str2: return 1.0
    if not str1 or not str2: return 0.0
    return Levenshtein.normalized_similarity(str1, str2)

# Feature: f6_name_levenshtein
row['f6_name_levenshtein'] = levenshtein_ratio(r1['name_clean'], r2['name_clean'])
```

**Catches:** "Mcdonald's" vs "McDonalds" (one edit: apostrophe removal)

#### Feature 7: Jaro-Winkler Similarity
A specialized edit-distance metric that **weights prefix matches more heavily**. Great for names where the beginning is usually more discriminating.

$$\text{JaroWinkler}(s1, s2) = \text{Jaro}(s1, s2) + p \cdot \text{prefix\_len} \cdot (1 - \text{Jaro}(s1, s2))$$

```python
from rapidfuzz.distance import JaroWinkler

# Feature: f7_name_jarowinkler
row['f7_name_jarowinkler'] = JaroWinkler.normalized_similarity(r1['name_clean'], r2['name_clean'])
```

**Catches:** "Willam" vs "William" — a common OCR/transcription error.

#### Feature 8: Address Levenshtein Ratio
Same Levenshtein ratio applied to the full cleaned address strings.

```python
# Feature: f8_addr_levenshtein
row['f8_addr_levenshtein'] = levenshtein_ratio(r1['addr_clean'][:200], r2['addr_clean'][:200])
```

#### Feature 9: Token Sort Ratio (Order-Invariant Name Matching)
Sorts the tokens alphabetically before comparing — catches word-order transpositions.

"Marketing Ram India" vs "Ram Marketing India" → after sorting: "India Marketing Ram" == "India Marketing Ram" → ratio = 1.0

```python
def token_sort_ratio(str1: str, str2: str) -> float:
    s1_sorted = ' '.join(sorted(str1.split()))
    s2_sorted = ' '.join(sorted(str2.split()))
    return Levenshtein.normalized_similarity(s1_sorted, s2_sorted)

# Feature: f9_name_token_sort_ratio
row['f9_name_token_sort_ratio'] = token_sort_ratio(r1['name_clean'], r2['name_clean'])
```

### 8.4 Category C: TF-IDF / Information-Theoretic Features

#### Feature 10 & 11: TF-IDF Cosine Similarity (Name & Address)

**Concept:** A word appearing in many business names (like "Services", "India", "Company") is a weak signal. A word appearing in only 2 names (like "Orelee") is a strong signal. TF-IDF weights rare words higher.

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\frac{N}{\text{DF}(t)}$$

Where TF = term frequency in document, N = total documents, DF = document frequency.

**Implementation:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fit on a SAMPLE of S1 + S2 names (not the full 7M — too slow)
all_names = list(s1_sample['name_clean']) + list(s2_sample['name_clean'])
tfidf_name = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
tfidf_name.fit(all_names)

# For a pair:
def tfidf_cosine(str1: str, str2: str, vectorizer) -> float:
    v1 = vectorizer.transform([str1])
    v2 = vectorizer.transform([str2])
    return float(cosine_similarity(v1, v2)[0, 0])

# Feature: f10_name_tfidf_cosine
row['f10_name_tfidf_cosine'] = tfidf_cosine(r1['name_clean'], r2['name_clean'], tfidf_name)

# Feature: f11_addr_tfidf_cosine
row['f11_addr_tfidf_cosine'] = tfidf_cosine(r1['addr_clean'], r2['addr_clean'], tfidf_addr)
```

> [!NOTE]
> Fit TF-IDF on the **full vocabulary sample** but **not on the test pairs' labels**. This prevents data leakage.

### 8.5 Category D: Structural / Data Quality Flag Features

These are binary or categorical features encoding metadata about the pair.

#### Feature 12: Script Mismatch Flag
```python
# Feature: f12_script_mismatch
row['f12_script_mismatch'] = int(r1['is_non_latin'] != r2['is_non_latin'])
```

**Value:** 1 if one name is Latin and the other is non-Latin (Hindi/Tamil/etc.), 0 otherwise.  
**Meaning:** When this is 1, the model knows that all name similarity features are unreliable and must rely more on address features.

#### Feature 13: Address Empty Flag
```python
# Feature: f13_addr_empty
# 0 = both have addresses, 1 = one or both are empty
row['f13_addr_empty'] = int(not r1['addr_clean'] or not r2['addr_clean'])
```

**Value:** 1 if either address is blank (after cleaning).  
**Meaning:** When address is empty, the model must rely solely on name features. If name is also non-Latin, the pair is likely unmatchable.

#### Feature 14: Domain Name Flag
```python
# Feature: f14_is_domain
row['f14_is_domain'] = int(r1['is_domain'] or r2['is_domain'])
```

**Value:** 1 if either record's name appears to be a website domain.  
**Meaning:** When this is 1, word-token Jaccard is unreliable. The model should weight character n-gram similarity more.

#### Feature 15: Country Match Flag
```python
# Feature: f15_country_match
row['f15_country_match'] = int(r1['s1_country'] == r2['s2_country'])
```

**Value:** 1 if same country, 0 if different countries.  
**Meaning:** From EDA, 100% of true matches are within the same country. If this flag is 0, it's almost certainly a non-match.

### 8.6 Final Feature Matrix

After processing all 50,000 pairs:

```
feature_matrix.csv (50,000 rows × 16 columns)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
f1_name_jaccard         f2_addr_jaccard          f3_name_dice
f4_addr_numeric_overlap f5_name_char3gram         f6_name_levenshtein
f7_name_jarowinkler     f8_addr_levenshtein       f9_name_token_sort_ratio
f10_name_tfidf_cosine   f11_addr_tfidf_cosine     f12_script_mismatch
f13_addr_empty          f14_is_domain             f15_country_match
label
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Feature Correlation Analysis (EDA Part 2):**
```python
import seaborn as sns
corr = feature_df.drop('label', axis=1).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
```
This heatmap shows which features are redundant (high correlation with each other) and which are uniquely informative.

---

## 9. Phase 6 — ML Model Comparison (The Core)

**Who:** Sumit Singh (240344) + Dhruv Sharma (240444) | **Time Estimate:** 10-12 hours  
**Deliverable:** `notebooks/04_Model_Comparison.ipynb` + results tables

### 9.1 Train-Test Split Strategy

**We use Stratified 5-Fold Cross-Validation** as our primary evaluation method.

```python
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score

X = feature_df.drop('label', axis=1).values
y = feature_df['label'].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Why Stratified? 
# Because our dataset has 50/50 label balance that we want preserved in each fold.
# Without stratification, a fold might accidentally get 60/40 split, biasing results.
```

**Additionally:** Create a fixed hold-out test set (20% of data) for final evaluation that is **never touched** during model selection:
```python
from sklearn.model_selection import train_test_split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
```

### 9.2 Evaluation Metrics — Why F₀.₅ and Not Just Accuracy?

**Accuracy** is misleading when the goal is precision-sensitive. For entity resolution:
- **False Positive (FP):** Saying two different businesses match — merges wrong records into one entity, causing real business damage.
- **False Negative (FN):** Missing a true match — a less serious error.

Therefore, we weight **Precision 2× over Recall** using F₀.₅:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

With $\beta = 0.5$: Precision is weighted 4× as much as recall in the harmonic mean.

**All metrics we report for every model:**
1. **Precision** = TP / (TP + FP) — how reliable are our positive predictions?
2. **Recall** = TP / (TP + FN) — what fraction of true matches did we find?
3. **F₁ Score** = Harmonic mean of Precision and Recall (balanced).
4. **F₀.₅ Score** = Precision-weighted F-score (our primary metric).
5. **AUC-ROC** = Area Under the ROC Curve (threshold-independent performance).
6. **Confusion Matrix** — raw TP/TN/FP/FN counts.

```python
scoring = {
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'f_beta': make_scorer(fbeta_score, beta=0.5),
    'roc_auc': 'roc_auc'
}
```

### 9.3 The Six Models — What They Are and Why They're Different

#### Model 1: Logistic Regression (The Baseline)

**What it is:** Fits a linear decision boundary in feature space. Each feature gets a weight $w_i$:

$$P(y=1) = \sigma\left(\sum_{i=1}^{15} w_i \cdot x_i + b\right) = \frac{1}{1 + e^{-z}}$$

**What it teaches us:**
- The weight vector tells us directly which features the model thinks are most important.
- If Logistic Regression performs nearly as well as complex models, our features are inherently linearly separable.

**Regularization variants to compare:**
- `LogisticRegression(penalty='l2', C=1.0)` — Ridge (penalizes large weights, keeps all features)
- `LogisticRegression(penalty='l1', C=1.0)` — Lasso (drives irrelevant feature weights to exactly zero)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Logistic Regression needs feature scaling!
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42))
])
```

> [!NOTE]
> **Why do we scale for Logistic Regression?** Features like `name_jaccard` are in [0,1] but `addr_tfidf_cosine` might also be [0,1] but with different variance. Scaling ensures no feature dominates just due to scale difference.

**Bias-Variance:** Logistic Regression is a **high-bias, low-variance** model. It may underfit if the decision boundary is non-linear.

#### Model 2: K-Nearest Neighbors (KNN)

**What it is:** For a new pair, find the K most similar training pairs (in feature space) and vote on their labels.

$$\hat{y} = \text{mode}\left(\{y_1, y_2, ..., y_K\}\right) \text{ where } x_1, ..., x_K \text{ are K nearest training points}$$

**What it teaches us:**
- KNN is **non-parametric** — it doesn't learn explicit parameters. It's essentially the entire training dataset.
- Very sensitive to the choice of K and the distance metric.
- Can capture highly non-linear decision boundaries if K is small.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# KNN also needs scaling (Euclidean distance is scale-sensitive)
knn_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
])
```

**Experiment:** Test K = 3, 5, 7, 11 and compare. Plot K vs. F₀.₅ score.

**Bias-Variance:** 
- Small K (K=1): Low bias, high variance (overfitting to training data).
- Large K: High bias, low variance (underfitting, votes of many unlike neighbors).

**Expected limitation:** KNN is slow at prediction (must compute distance to all training points). This illustrates the **scalability-accuracy tradeoff** — a critical ML concept.

#### Model 3: Support Vector Machine (SVM)

**What it is:** Finds the **maximum-margin hyperplane** that separates positive and negative pairs.

$$\text{Maximize} \frac{2}{||w||}$$  subject to $y_i(w \cdot x_i + b) \geq 1$ for all training points.

**Kernel trick:** SVMs can implicitly project data into higher-dimensional space via the kernel function, enabling non-linear boundaries:
- `kernel='linear'`: Linear boundary (like Logistic Regression but maximum-margin).
- `kernel='rbf'` (Radial Basis Function): Circular, non-linear boundaries in feature space.

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

svm_linear = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(kernel='linear', C=1.0, probability=True, random_state=42))
])

svm_rbf = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
])
```

**What it teaches us:**
- Compare `linear` vs `rbf` kernel performance to understand if our feature space is linearly separable.
- The regularization parameter `C` controls the bias-variance tradeoff: large C → low bias/high variance (may overfit), small C → high bias/low variance.

#### Model 4: Decision Tree

**What it is:** Recursively splits the feature space using axis-aligned hyperplanes, creating if-then-else rules.

**Example of a learned decision tree rule:**
```
Is addr_jaccard > 0.5?
├── YES → Is country_match = 1?
│         ├── YES → PREDICT MATCH (1)
│         └── NO  → PREDICT NO MATCH (0)
└── NO  → Is script_mismatch = 1?
          ├── YES → PREDICT NO MATCH (0) 
          └── NO  → Is name_jaccard > 0.3?
                    ├── YES → PREDICT MATCH (1)
                    └── NO  → PREDICT NO MATCH (0)
```

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

dt = DecisionTreeClassifier(
    criterion='gini',           # Gini impurity vs Entropy — experiment with both
    max_depth=6,                # Prevents overfitting
    min_samples_leaf=50,        # At least 50 samples in each leaf
    random_state=42
)
```

**What it teaches us:**
- **Visualize the actual tree** (critical for professor presentation — shows the model is interpretable).
- Compare `criterion='gini'` vs `criterion='entropy'`.
- Plot Depth (max_depth) vs F₀.₅ score to show overfitting behavior.
- Decision trees have **high variance** — small changes in training data can produce very different trees. This motivates ensemble methods!

```python
# Visualize the tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=feature_names, class_names=['No Match', 'Match'],
          filled=True, max_depth=4)
plt.savefig("results/plots/decision_tree.png", dpi=150)
```

#### Model 5: Random Forest (Bagging Ensemble)

**What it is:** Train **N independent Decision Trees** on **N random bootstrap samples** of the training data, each tree also using a random subset of features. Final prediction = majority vote across all N trees.

**Why random + bagging reduces variance:**
- Each tree is trained on a different data sample → they make different errors.
- Since errors are independent, they cancel out when averaged.
- This is the **Wisdom of the Crowd** effect in ML.

$$\hat{y}_{\text{RF}} = \text{mode}(\hat{y}_1, \hat{y}_2, ..., \hat{y}_N)$$

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,       # 100 trees
    max_depth=10,
    max_features='sqrt',    # Each tree sees sqrt(15) ≈ 4 features at each split
    n_jobs=-1,              # Use all CPU cores
    random_state=42
)
```

**What it teaches us:**
- Compare against single Decision Tree to quantify the **variance reduction** from bagging.
- Feature Importance via Gini: Average decrease in impurity for each feature across all trees.
- More trees doesn't always mean better — plot `n_estimators` vs F₀.₅ to find the saturation point.

#### Model 6: Gradient Boosting (Boosting Ensemble)

**What it is:** Train **N trees sequentially**, where each new tree **specifically fixes the errors** of the previous trees.

```
Tree 1: Predict on all data → has errors
Tree 2: Focus on the pairs Tree 1 got WRONG → reduce those errors
Tree 3: Focus on the errors that remain after Tree 1 and 2 ...
...
Final prediction = weighted sum of all tree predictions
```

The mathematical update at each step:
$$F_{m+1}(x) = F_m(x) + \alpha \cdot h_m(x)$$

Where $h_m$ is the new tree trained on **residuals** (errors) of $F_m$, and $\alpha$ is the learning rate.

```python
from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,      # Step size for each new tree's correction
    max_depth=4,            # Keep trees shallow to prevent overfitting
    subsample=0.8,          # Stochastic gradient boosting
    random_state=42
)
```

**What it teaches us:**
- Boosting typically outperforms Bagging (Random Forest) on structured/tabular data.
- Much more prone to overfitting — requires careful tuning of `n_estimators` and `learning_rate`.
- Plot Training vs. Validation F₀.₅ across boosting rounds to show the **learning curve and overfitting point**.

### 9.4 Comparative Results Table

After running all 6 models with 5-fold cross-validation, populate this exact table:

| Model | Precision (CV) | Recall (CV) | F₁ (CV) | **F₀.₅ (CV)** | AUC-ROC (CV) | Training Time |
|---|---|---|---|---|---|---|
| Logistic Regression (L2) | | | | | | |
| Logistic Regression (L1) | | | | | | |
| KNN (k=5) | | | | | | |
| SVM (Linear) | | | | | | |
| SVM (RBF) | | | | | | |
| Decision Tree (depth=6) | | | | | | |
| Random Forest (100 trees) | | | | | | |
| Gradient Boosting | | | | | | |

---

## 10. Phase 7 — Hyperparameter Tuning & Cross-Validation

**Who:** Sumit Singh (240344) | **Time Estimate:** 6-8 hours  
**Deliverable:** `notebooks/05_Hyperparameter_Tuning.ipynb`

### 10.1 What Is Hyperparameter Tuning?

A hyperparameter is a setting you configure **before** training (not learned from data). Examples:
- Decision Tree: `max_depth`, `min_samples_leaf`
- SVM: `C`, `gamma`, `kernel`
- Random Forest: `n_estimators`, `max_depth`, `max_features`
- Gradient Boosting: `n_estimators`, `learning_rate`, `max_depth`

**Goal:** Find the combination of hyperparameters that maximizes F₀.₅ on the validation set without overfitting.

### 10.2 GridSearchCV — Systematic Hyperparameter Search

```python
from sklearn.model_selection import GridSearchCV

# Example: Tuning Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 8, 10, None],
    'max_features': ['sqrt', 'log2', 0.5],
    'min_samples_leaf': [1, 5, 20]
}

gs = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,                           # Inner 5-fold CV
    scoring='f1',                   # Optimize for F₁
    n_jobs=-1,                      # Parallelize
    verbose=1
)
gs.fit(X_trainval, y_trainval)
print(f"Best params: {gs.best_params_}")
print(f"Best CV F₁: {gs.best_score_:.4f}")
```

> [!TIP]
> **Nested Cross-Validation for Unbiased Evaluation:** We use a 5-fold **outer** CV for unbiased performance estimation and a 3-fold **inner** CV within GridSearchCV for hyperparameter selection. This prevents the "double-dipping" issue where the same validation data is used for both model selection AND performance reporting.

### 10.3 Key Experiments Per Model

**Logistic Regression:**
- Vary `C` ∈ {0.001, 0.01, 0.1, 1, 10, 100}
- Plot C vs. F₁ — observe underfitting at small C, potential overfitting at large C.
- Compare L1 vs L2 — which features does L1 zero out?

**Decision Tree:**
- Vary `max_depth` ∈ {2, 4, 6, 8, 10, 15, None}
- **Critical plot:** Training F₀.₅ vs Validation F₀.₅ vs Depth — this is the **Overfitting / Bias-Variance Tradeoff** curve. The professor **will** ask about this.
- Observe: At depth=2, both curves are low (underfitting). At depth=15, training F₀.₅ ≈ 1.0 but validation drops (overfitting).

**Random Forest:**
- Vary `n_estimators` ∈ {10, 25, 50, 100, 200}
- Plot: Trees vs. Val F₀.₅ — shows convergence (diminishing returns)
- Vary `max_features` ∈ {'sqrt', 'log2', 0.5, 1.0}

**Gradient Boosting:**
- Key tradeoff: `n_estimators` vs `learning_rate` (more rounds + smaller rate = better but slower)
- Plot: Boosting rounds vs. Train/Val score — shows the optimal stopping point.
- Concept: **Early Stopping** — stop adding trees when validation score stops improving.

### 10.4 Learning Curves (Bias-Variance Diagnosis)

For the best model, generate learning curves:

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_trainval, y_trainval,
    cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='f1'
)

# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training F₁')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation F₁')
plt.xlabel('Training Set Size')
plt.ylabel('F₁ Score')
plt.title('Learning Curve — Gradient Boosting')
plt.legend()
```

**What to look for:**
- **High bias (underfitting):** Both curves converge at a low value → model is too simple, need more features or a more complex model.
- **High variance (overfitting):** Large gap between training and validation curves → model is too complex, need regularization or more data.

---

## 11. Phase 8 — Ensemble Methods

**Who:** Dhruv Sharma (240444) | **Time Estimate:** 4-6 hours  
**Deliverable:** Section in `notebooks/04_Model_Comparison.ipynb`

### 11.1 Why Ensemble? The Diversity Principle

No single model is best on all examples:
- Logistic Regression handles linearly separable easy cases well.
- Decision Tree handles complex multi-feature interactions.
- SVM excels in high-dimensional spaces near the margin.

By combining them, errors made by different models are **uncorrelated**, and the combined model achieves higher accuracy than any single model.

### 11.2 Soft Voting Ensemble

Combine the probability predictions of multiple models:

$$P_{\text{ensemble}}(y=1) = \frac{1}{M} \sum_{m=1}^M P_m(y=1)$$

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('lr', best_logistic_regression),
        ('svm', best_svm_rbf),
        ('rf', best_random_forest),
        ('gbm', best_gradient_boosting)
    ],
    voting='soft'    # Use probability averaging
)

cv_scores = cross_validate(ensemble, X_trainval, y_trainval, cv=5, scoring=scoring)
```

### 11.3 Stacking Ensemble (Meta-Learner)

A more sophisticated approach: train a "meta-model" on the **out-of-fold predictions** of base models.

```python
from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(
    estimators=[
        ('lr', best_lr),
        ('rf', best_rf),
        ('gbm', best_gbm)
    ],
    final_estimator=LogisticRegression(C=1.0),  # Meta-learner
    cv=5,  # Out-of-fold predictions for training the meta-learner
    passthrough=False
)
```

**Concept:** The meta-learner learns which base model to trust more for different types of pairs.

### 11.4 Final Ensemble Comparison Table

| Method | F₀.₅ (CV) | F₁ (CV) | Improvement Over Best Single |
|---|---|---|---|
| Best Single Model | | | — |
| Soft Voting (3 models) | | | |
| Soft Voting (all 4) | | | |
| Stacking (Logistic Meta) | | | |

---

## 12. Phase 9 — Model Interpretability & Error Analysis

**Who:** Dhruv Sharma (240444) | **Time Estimate:** 4 hours

> [!NOTE]
> This section is what shows the professor you **understand** your results, not just that you ran code. It directly addresses CO3.

### 12.1 Feature Importance Analysis

**For Logistic Regression:**
```python
importance = pd.DataFrame({
    'feature': feature_names,
    'weight': best_lr.named_steps['model'].coef_[0]
})
importance['abs_weight'] = importance['weight'].abs()
importance.sort_values('abs_weight', ascending=False, inplace=True)
# Bar plot of weights
```

**For Random Forest and Gradient Boosting:**
```python
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)
# Bar plot of Gini importances
```

**Expected insight:** `f2_addr_jaccard`, `f4_addr_numeric_overlap`, and `f10_name_tfidf_cosine` will likely be the top features. This validates our EDA finding that addresses are more discriminating than names.

### 12.2 Error Analysis — What Does The Best Model Get Wrong?

```python
# Get predictions on the test set
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# False Positives: Different businesses we incorrectly called "Match"
false_positives = test_df[(y_pred == 1) & (y_test == 0)]
# False Negatives: Same businesses we incorrectly called "No Match"  
false_negatives = test_df[(y_pred == 0) & (y_test == 1)]
```

**Analysis questions:**
1. What is the `script_mismatch` rate among False Negatives? (Probably high — script-mismatch pairs are hard.)
2. What is the `is_domain` rate among False Positives? (Domain pairs might fool name similarity.)
3. What is the average `name_jaccard` of False Positives? (Probably high — same-name, different-address failures.)

This analysis directly motivates the features we added and shows the professor that you understand *where* the model struggles and *why*.

### 12.3 ROC Curve Comparison Plot

```python
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))
for name, model in best_models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — All Models')
plt.legend()
plt.savefig("results/plots/roc_comparison.png", dpi=150)
```

### 12.4 Confusion Matrix for Each Model

```python
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ax, (name, model) in zip(axes.flat, best_models.items()):
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=['No Match', 'Match'],
        ax=ax, colorbar=False
    )
    ax.set_title(name)
plt.savefig("results/plots/confusion_matrices.png", dpi=150)
```

---

## 13. Phase 10 — Final Report & Deliverables

**Who:** All members | **Time Estimate:** 8-10 hours

### 13.1 Report Template — Section by Section (from Your docx)

The submitted report must follow this exact structure from `Final_Project_Report_Template.docx`:

| Section | Pages | What It Must Contain |
|---|---|---|
| **Title Page** | 1 | Project title, member names & enrollment numbers, Dept. CSE, BML Munjal, November 2026 |
| **Declaration** | 1 | Signed statement that work is original |
| **Introduction** | 1 | Context — what is entity resolution, why is it important in industry (use 2-3 real examples: banking KYC, government databases, e-commerce) |
| **Problem Statement** | 2-3 | Clear formal definition. Input (a pair of records), Output (binary prediction), metric (F₀.₅), dataset description |
| **Dataset Description** | 1 | Table with file names, sizes, row counts, schema. EDA summary statistics. Data quality issues found. |
| **Literature Review** | 2-3 | 5+ papers on entity resolution / record linkage. Must include gap analysis — what existing methods don't solve that we address. |
| **Methodology** | 8-16 | **THE BIGGEST SECTION.** Must cover: preprocessing pipeline, feature engineering (all 15 features with justification), each model (what it is, math, why chosen), evaluation methodology (cross-validation, metrics), hyperparameter tuning |
| **Technology Stack** | 1 | Python 3.11, scikit-learn 1.3.2, pandas, numpy, rapidfuzz, matplotlib, seaborn, jupyter. Brief reason for each. |
| **Experimental Results** | 7-8 | Tables + figures. Cross-validation results. Confusion matrices. ROC curves. Feature importance. Hyperparameter tuning plots. Error analysis. |
| **Team Responsibility Matrix** | 1 | Table with all 4 members and their contributions to each task |
| **Conclusions** | 1 | What we found, what surprised us, limitations, future work |
| **References** | 1 | At minimum 8 cited works in IEEE format |
| **Appendix** | — | Key code snippets (not all code) |

---

## 14. Team Responsibility Matrix

| Task | Ananshi (240377) | Aditya (240368) | Sumit (240344) | Dhruv (240444) |
|---|---|---|---|---|
| **Data Preprocessing & EDA** | **Lead** | Support | | |
| **Benchmark Dataset Construction** | **Lead** | Support | | |
| **Feature Engineering** | Support | **Lead** | Support | |
| **Logistic Regression, SVM, KNN** | | Support | **Lead** | |
| **Decision Tree, Random Forest, GBM** | | | Support | **Lead** |
| **Hyperparameter Tuning** | | | **Lead** | Support |
| **Ensemble Methods** | | | Support | **Lead** |
| **Model Interpretability & Error Analysis** | | | | **Lead** |
| **ROC/PR Curves, Visualizations** | Support | Support | **Lead** | Support |
| **Report Writing** | **Lead** | Sections 4-5 | Sections 6-7 | Sections 7-8 |
| **Code Review & Integration** | Support | **Lead** | Support | Support |

---

## 15. Course Outcome Mapping

| Course Outcome | Requirement | How Our Project Covers It |
|---|---|---|
| **CO1:** Feature engineering & classification | Apply ML algorithms using feature engineering | 15 engineered features across 4 categories: Jaccard/Dice/Levenshtein/TF-IDF/Flags |
| **CO2:** Develop models for real-world problems | Build models for real-world data | 6 model types on real industry dataset (2.2M+ businesses) |
| **CO3:** Evaluate models on performance metrics | Use metrics: Precision, Recall, Confusion Matrix, ROC, CV | Full evaluation suite: F₀.₅, F₁, AUC-ROC, 5-fold CV, hyperparameter tuning |

---

## 16. What Gets Graded — Report Template Checklist

The professor evaluates 40% weight on these areas from the template:

- [ ] **Introduction** — Context, motivation, real-world impact
- [ ] **Problem Statement** — Formally defined, input/output/metric clear
- [ ] **Dataset Description** — Exact statistics, quality issues documented
- [ ] **Literature Review** — 5+ papers cited, gaps identified
- [ ] **Methodology (Preprocessing)** — Each step justified with EDA findings
- [ ] **Methodology (Feature Engineering)** — Each feature has mathematical definition + justification
- [ ] **Methodology (Models)** — Each model explained conceptually + mathematically
- [ ] **Experimental Results** — Tables with real numbers (not empty), all 8 models reported
- [ ] **Confusion Matrices** — Shown for all models
- [ ] **ROC Curves** — Multi-model comparison plot
- [ ] **Hyperparameter Tuning** — At minimum Depth-vs-Score for DT, C-vs-Score for LogReg
- [ ] **Cross-Validation** — 5-fold CV explained and applied
- [ ] **Feature Importance** — Bar chart for RF + GBM, coefficients for LogReg
- [ ] **Error Analysis** — What does the best model get wrong and why?
- [ ] **Team Matrix** — Filled with specific contributions, not generic
- [ ] **Conclusions** — Specific findings, limitations, future work
- [ ] **References** — At minimum 8 IEEE-formatted citations
- [ ] **Appendix** — Key code snippets

---

## 17. Research Gaps & Literature Review

### 5 Core Papers to Cite

1. **Christen, Peter (2012). "Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection."** Springer.
   - *Foundational textbook on entity resolution. Establishes the blocking-then-classification pipeline we use.*
   - *Gap addressed by us:* Christen focuses on traditional string similarity; we incorporate TF-IDF weighting for rare-token advantage.

2. **Papadakis et al. (2020). "Blocking and Filtering Techniques for Entity Resolution: A Survey." ACM Computing Surveys.**
   - *Comprehensive survey of candidate generation (blocking) strategies.*
   - *Gap addressed by us:* We show that for academic-scale projects, careful negative sampling can replace complex blocking, still yielding comparable classifier training.

3. **Mudgal et al. (2018). "Deep Learning for Entity Matching: A Design Space Exploration." SIGMOD.**
   - *Shows deep learning models can outperform hand-crafted features for ER.*
   - *Gap addressed by us:* Deep learning requires 100K+ labeled pairs and significant GPU compute. Our approach using interpretable features achieves comparable F₁ in a fraction of the compute budget — critical for resource-constrained environments.*

4. **Li et al. (2020). "Deep Entity Matching with Pre-Trained Language Models." VLDB (Ditto).**
   - *State-of-the-art ER using BERT/transformers.*
   - *Gap addressed by us:* Transformers are language-specific and fail on cross-script pairs (Hindi ↔ English) — a core challenge in our dataset. Our structural features (address numeric overlap, script flags) are language-agnostic.*

5. **Jurafsky, D. & Martin, J.H. (2023). "Speech and Language Processing." Chapter 2 — Regular Expressions, Text Normalization.**
   - *Authoritative reference for text normalization pipeline (lowercase, punctuation, tokenization).*
   - *Forms the theoretical basis for our Phase 4 preprocessing.*

### Key Gap We Address

> **The core research gap:** Most entity resolution literature assumes both records are in the same language and script. In real-world developing economies (India, Southeast Asia), business records frequently exist in native scripts (Devanagari, Tamil) while reference databases are entirely in English. Our project systematically identifies this cross-script challenge through EDA and addresses it through language-agnostic structural features (address number matching, script mismatch flags), showing that transformer models are not required to handle cross-script entity resolution at acceptable precision levels.

---

## 18. Final Deliverables Summary

```
PROJECT DELIVERABLES (Due: Last 2 Weeks of November 2026)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 1. Final Project Report (Word Document, ~25-30 pages)
   └── Follows the exact template structure from Final_Project_Report_Template.docx

💻 2. Source Code (Python files + Jupyter Notebooks)
   ├── src/preprocess.py
   ├── src/features.py
   ├── src/models.py
   ├── src/evaluate.py
   └── notebooks/01_EDA.ipynb through 06_Ensemble.ipynb

📊 3. Results Package
   ├── results/tables/model_comparison_table.csv
   ├── results/plots/roc_comparison.png
   ├── results/plots/confusion_matrices.png
   ├── results/plots/feature_importance.png
   └── results/plots/hyperparameter_tuning/*.png

📦 4. Processed Dataset (for reproducibility)
   └── data/processed/benchmark_pairs.csv (50K pairs with label)

🎤 5. Presentation + Viva
   └── 10-15 slide deck covering all methodology, results, and interpretations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Appendix A: Implementation Timeline

```
Week 1 (Nov 15-21):
  Day 1:   Setup environment + verify dataset loads correctly
  Day 2-3: Phase 2 — Benchmark dataset construction (Ananshi)
            Phase 3 EDA Part 1 — Dataset-level stats (Ananshi)
  Day 4-5: Phase 3 EDA Part 2 — Pair-level feature distribution analysis (Ananshi)
            Phase 4 — Preprocessing pipeline implementation (Ananshi)
  Day 6-7: Phase 5 — Feature engineering all 15 features (Aditya)
            Feature matrix generation + correlation analysis (Aditya)

Week 2 (Nov 22-28):
  Day 8-9: Phase 6 — Logistic Regression + SVM + KNN (Sumit)
  Day 10-11: Phase 6 — Decision Tree + Random Forest + GBM (Dhruv)
  Day 12-13: Phase 7 — Hyperparameter tuning (Sumit)
              Phase 8 — Ensemble methods (Dhruv)
  Day 14: Phase 9 — Error analysis + interpretability (Dhruv)
           All plots and figures finalized

Week 3 (Nov 29 - Submission):
  Day 15-16: Report writing — Methodology + Results sections (All)
  Day 17-18: Report writing — Introduction, Literature Review, Conclusions (Ananshi leads)
  Day 19:    Presentation slides prepared
  Day 20:    Final review + professor submission
```

---

## Appendix B: Commands to Reproduce Everything

```bash
# 1. Setup
git clone <your-repo-url>
cd ml_project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Build benchmark dataset
python src/make_benchmark.py \
  --s1 data/raw/train_source1.tsv \
  --s2 data/raw/train_source2.tsv \
  --gt data/raw/train_ground_truth.tsv \
  --n_positives 25000 \
  --n_negatives 25000 \
  --seed 42 \
  --output data/processed/benchmark_pairs.csv

# 3. Extract features
python src/extract_features.py \
  --pairs data/processed/benchmark_pairs.csv \
  --output data/processed/feature_matrix.csv

# 4. Run model comparison
python src/run_experiments.py \
  --features data/processed/feature_matrix.csv \
  --results results/tables/model_comparison.csv

# 5. Launch notebooks (for detailed analysis + plots)
jupyter lab
```
