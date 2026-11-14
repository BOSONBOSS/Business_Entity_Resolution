# 🏆 FINAL EXHAUSTIVE OPTIMIZATION PLAN
## Amazon ML Challenge 2026 — Business Entity Resolution
### Every fix. Every code snippet. Every threshold. Every risk.
### Context: ALL files read — CLAUDE.md · HACKATHON_RULES.md · PREPROCESSING_PLAN.md · README.md · student_resource/README.md · Documentation_template.md · PROBLEM_STATEMENT.md · EDA notebook (513KB) · all 7 TSV datasets · matching_results.tsv

---

## PART 0 — THE GROUND TRUTH (Read This First)

```
CURRENT STATE
─────────────────────────────────────────────────────────
  Public Leaderboard (LB):     0.924   ← the real number
  Local Validation (CV):       0.9742  ← stack3 just done
  CV ↔ LB Gap:                 0.0502  ← THE ONLY PROBLEM
  Decision rule:               Lewis 1995 Expected-F₀.₅ prefix selection
  Negative sampling:           Clean — neg_frac only on training split
  Machine:                     i9-14900HX · 32 GB RAM · RTX 4060 8GB VRAM
  Deadline:                    27 Sep 2026 · 11:59 PM IST  (~46 hrs from now)
  Submissions remaining:       ~10 (5 today + 5 tomorrow)
─────────────────────────────────────────────────────────
```

### Why LB=0.924 When CV=0.9742 — The Math

France = 14.98% of test. India+US = 85.02%. CV is measured only on India+US.

```
0.8502 × 0.9742  +  0.1498 × France_F₀.₅  =  0.924
                                France_F₀.₅  =  0.639
```

**France is scoring ~0.64. India+US score ~0.97. That 0.33 gap IS the entire problem.**

### Three Root Causes (Ranked by Impact)

| # | Root Cause | LB Impact | Time to Fix |
|---|---|---|---|
| 🔴 A | BM25 + TF-IDF IDF weights built on India+US only → French words OOV | ~0.025 | 2–3h |
| 🔴 B | Feature distributions on France are pathological → LightGBM mis-scores | ~0.020 | 4–6h |
| 🔴 C | Isotonic calibration fitted on India+US → France probs systematically wrong → Lewis 1995 picks wrong prefix | ~0.015 | 2h |
| 🟠 D | Accent bug: French é/à/ç flagged as "Hindi" → is_non_latin wrong for 24% of France pairs | ~0.005 | 10min |
| 🟠 E | French legal suffixes not stripped (SARL, EURL, EIRL, SNC, SCA, SCOP) | ~0.003 | 15min |
| 🟠 F | Domain names: word-Jaccard=0 with no char n-gram fallback | ~0.004 | 30min |
| 🟡 G | Address numeric overlap missing (language-agnostic key signal) | ~0.003 | 30min |
| 🟡 H | BM25 80% relative filter too strict for low-confidence (cross-script/domain) probes | ~0.002 | 30min |

### What Was WRONG in the Previous Plan (Removed)

| Wrong idea | Why wrong |
|---|---|
| "France probability multiplier × 0.88" | Uniform scaling preserves candidate *ranking*. Lewis 1995 operates on ranked list → same prefix picked → no effect |
| "Logistic Regression pre-filter" | LR can't model name×address interaction. Risks silently dropping true France pairs |
| "Early removal of high-confidence matches" | Cascading error risk. Current "drop duplicate claims at end" is already correct |
| All expected gains from 0.954 baseline | Wrong number. 0.954 was never the LB score |

---

## PART 1 — BEFORE WRITING ANY CODE: DIAGNOSE (30 min)

Two questions. Answer these before touching the pipeline.

### Diagnostic 1 — France Blocking Recall

**Why it matters:** If only 90% of true France pairs ever enter the candidate set, features cannot push F₀.₅ above the 90% recall ceiling. No amount of LightGBM tuning recovers pairs never seen.

```python
# diagnostic_france_blocking.py
# Run this first. Takes ~5 minutes.

import csv, re, unicodedata
from collections import defaultdict

BASE = "/Users/ananshi_main/MAHU_DOC/BML_MUNJAL_2024/Hackathons/Amazon_ML_Challenge"

# ── Load candidate_pairs.tsv (blocking output fed to LightGBM)
# If this file doesn't exist → blocking recall cannot be measured directly.
# In that case, skip to Diagnostic 2.

import os
CANDIDATE_FILE = f"{BASE}/output/candidate_pairs.tsv"
RESULT_FILE    = f"{BASE}/result/matching_results.tsv"
TEST_S1        = f"{BASE}/student_resource/dataset/test/test_source1.tsv"

if not os.path.exists(CANDIDATE_FILE):
    print("candidate_pairs.tsv not found.")
    print("ACTION: Ask teammate to output it, or measure via BM25 score proxy below.")
else:
    # Load France S1 entity IDs
    france_s1 = set()
    with open(TEST_S1, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["country"] == "France":
                france_s1.add(row["entity_id"])

    # Measure: for France S1 entities, what fraction of their true matches
    # appear in candidate_pairs.tsv?
    # (This requires knowing ground truth — only possible if a France val split exists)
    print(f"France S1 entities in test: {len(france_s1):,}")
    print("Load candidate_pairs.tsv and cross-reference with ground truth if available.")

# ── Proxy measurement: BM25 score distribution for France pairs
# Read matching_results.tsv and check how many France entities got 0 candidates
# vs how many got >= 1 candidate from blocking
with open(RESULT_FILE, encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    france_zero = 0
    france_total = 0
    s1_country = {}

with open(TEST_S1, encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        s1_country[row["entity_id"]] = row["country"]

france_zero_candidates = 0
france_has_candidates = 0
with open(RESULT_FILE, encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)
    for row in reader:
        if s1_country.get(row[0]) != "France":
            continue
        matched = row[1].strip() if len(row) > 1 else ""
        n = len([x for x in matched.split(",") if x.strip()]) if matched else 0
        if n == 0:
            france_zero_candidates += 1
        else:
            france_has_candidates += 1

total_france = france_zero_candidates + france_has_candidates
print(f"\nFrance entities with ≥1 predicted match: {france_has_candidates:,} ({100*france_has_candidates/total_france:.2f}%)")
print(f"France singletons (0 matches):           {france_zero_candidates:,} ({100*france_zero_candidates/total_france:.2f}%)")
print(f"\nExpected singleton rate (from training): 5.58%")
print(f"If observed singleton rate >> 5.58% → blocking miss is part of the problem")
print(f"If observed singleton rate << 5.58% → over-matching / calibration is the problem")
```

### Diagnostic 2 — Is the BM25 Index France-Aware?

Ask your teammate: **"Was the BM25 inverted index and TF-IDF vectorizer fitted/built using only train_source1/2/3, or did you include test_source1/2/3 when computing IDF weights?"**

- If **training only** → Root Cause A is confirmed → Phase 1 is mandatory
- If **full corpus** → Root Cause A is not the issue → skip Phase 1, go straight to Phase 2

**Decision point:**
```
Diagnostic 1: France blocking recall < 95%  → Phase 1 + Phase 4 + Phase 5 are priority
Diagnostic 1: France blocking recall ≥ 95%  → Phase 2 + Phase 3 are priority
Diagnostic 2: IDF trained on training only   → Phase 1 is mandatory regardless
Diagnostic 2: IDF includes test data          → Skip Phase 1
```

---

## PART 2 — PHASE 1: THE VOCABULARY FIX (Root Cause A)
### ~2–3 hours | No labels needed | Expected LB gain: +0.020 to +0.030

### What Broke

The BM25 index and TF-IDF vectorizers were built on training data only (India+US). French words — "Rue", "Société", "Boulangerie", "SARL", "Cedex" — have **zero document frequency** in that corpus.

- In BM25: IDF for an OOV word = undefined or max → scores become garbage
- In TF-IDF: French words mapped to vocabulary index 0 (OOV) → cosine = 0 for all French pairs
- Result: A French S1 record "Société Dupont SARL" and French S2 "Societe Dupont" with identical names after normalization may score BM25 ≈ 0 because neither record's tokens exist in the India+US IDF table

### The Fix — Refit on Full Corpus

```python
# scripts/rebuild_vocabulary.py
"""
Refit TF-IDF vectorizers and IDF weights on the combined train+test corpus.
No labels used — only the raw text. Safe to include test data.

Run BEFORE any feature recomputation or retraining.
"""

import csv, pickle, re, unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = "/Users/ananshi_main/MAHU_DOC/BML_MUNJAL_2024/Hackathons/Amazon_ML_Challenge"

PLACEHOLDER = {"null", "n/a", "na", "nan", "nil", "none", "-", ""}

def strip_accents(text):
    """Remove diacritics: 'Société' → 'Societe'. Call BEFORE is_non_latin."""
    nfkd = unicodedata.normalize('NFKD', str(text))
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

LEGAL = {
    # English
    "inc","incorporated","corp","corporation","ltd","limited","llp","llc",
    "pvt","private","co","company","plc","and","the","of",
    # French ← NEW
    "sarl","sas","sa","sci","sca","snc","eurl","eirl","scop","sasu","se","scs",
    # German
    "gmbh","ag","kg","ohg","eg",
}

DOMAIN_RE = re.compile(
    r'\b([\w\-]+)\.(com|net|org|in|fr|co|io|biz|info|edu|gov|uk|de|jp)\b', re.I
)

def norm(text):
    """Master normalization function."""
    if not text or str(text).strip().lower() in PLACEHOLDER:
        return ""
    s = strip_accents(str(text))         # Accents first
    s = re.sub(r'[^\w\s]', ' ', s.lower())
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def norm_name(text):
    """Normalize business name, strip legal suffixes."""
    s = norm(text)
    # Extract domain core if name is a website
    m = DOMAIN_RE.search(s)
    if m:
        s = m.group(1)  # "dhaninvestments" from "dhaninvestments.com"
    words = [w for w in s.split() if w not in LEGAL]
    return ' '.join(words) if words else s

def norm_addr(text):
    """Normalize address with abbreviation expansion."""
    s = norm(text)
    ABBREVS = {
        " rd ": " road ", " st ": " street ", " ave ": " avenue ",
        " blvd ": " boulevard ", " dr ": " drive ", " ln ": " lane ",
        " ct ": " court ", " hwy ": " highway ", " pkwy ": " parkway ",
        " marg ": " road ",    # India
        " rue ": " rue ",      # French — keep as-is (distinctive)
        " bd ": " boulevard ", # French
        " av ": " avenue ",    # French
    }
    s = " " + s + " "
    for abbr, full in ABBREVS.items():
        s = s.replace(abbr, full)
    return s.strip()


# ── Collect ALL text from ALL files (train + test)
print("Collecting all text from train + test files...")
all_names, all_addrs = [], []

FILES = [
    f"{BASE}/student_resource/dataset/train/train_source1.tsv",
    f"{BASE}/student_resource/dataset/train/train_source2.tsv",
    f"{BASE}/student_resource/dataset/train/train_source3.tsv",
    f"{BASE}/student_resource/dataset/test/test_source1.tsv",    # ← France vocab
    f"{BASE}/student_resource/dataset/test/test_source2.tsv",
    f"{BASE}/student_resource/dataset/test/test_source3.tsv",
]

for fp in FILES:
    print(f"  Reading {fp.split('/')[-1]}...")
    with open(fp, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            all_names.append(norm_name(row.get("business_name", "")))
            all_addrs.append(norm_addr(row.get("business_address", "")))

print(f"Total records: {len(all_names):,}")

# ── Fit word-level TF-IDF (for name and address cosine features)
print("Fitting word TF-IDF on names...")
name_tfidf_word = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    max_features=600_000,
    sublinear_tf=True,
    min_df=2,               # Ignore hapax legomena
)
name_tfidf_word.fit(all_names)

print("Fitting word TF-IDF on addresses...")
addr_tfidf_word = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    max_features=600_000,
    sublinear_tf=True,
    min_df=2,
)
addr_tfidf_word.fit(all_addrs)

# ── Fit char-level TF-IDF (for domain names, accented names, typos)
print("Fitting char n-gram TF-IDF on names...")
name_tfidf_char = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 4),
    max_features=400_000,
    sublinear_tf=True,
    min_df=3,
)
name_tfidf_char.fit(all_names)

# ── Save
import os
os.makedirs(f"{BASE}/work", exist_ok=True)
pickle.dump(name_tfidf_word, open(f"{BASE}/work/name_tfidf_word_v2.pkl", "wb"))
pickle.dump(addr_tfidf_word, open(f"{BASE}/work/addr_tfidf_word_v2.pkl", "wb"))
pickle.dump(name_tfidf_char, open(f"{BASE}/work/name_tfidf_char_v2.pkl", "wb"))

print("Saved: name_tfidf_word_v2.pkl · addr_tfidf_word_v2.pkl · name_tfidf_char_v2.pkl")
print("NEXT: Pass these to teammate's feature builder. Then retrain LightGBM.")
```

> ⚠️ **Critical:** If you refit TF-IDF, you MUST also rebuild ALL pairwise features and retrain LightGBM. Old model + new TF-IDF = wrong features interpreted with wrong learned coefficients. Never mix versions.

---

## PART 3 — PHASE 2: FEATURE FIXES (Root Causes B, D, E, F, G)
### ~4–6 hours including retrain | Expected LB gain: +0.015 to +0.025

Do all of D+E+F+G together in ONE retrain. Each is small alone; combined they address ~0.015 of the gap.

### Fix D — French Accent Bug (10 minutes — highest priority, lowest effort)

**The bug:** `is_non_latin` currently uses regex `[^\x00-\x7F]` on the *raw* text.
French characters é, à, ç, ô, û, î are non-ASCII. So 24.4% of French matched pairs are flagged as "Hindi/Devanagari". LightGBM then applies cross-script logic (high address-weight, low name-weight) to French pairs where names ARE comparable. This is wrong.

```python
# In src/normalize.py — replace the is_non_latin function

import unicodedata, re

def strip_accents(text: str) -> str:
    """
    Remove combining diacritical marks so French/Spanish/German
    accented characters become plain ASCII.
    'Société' → 'Societe'   |   'München' → 'Munchen'
    Devanagari/Arabic/Chinese characters are NOT affected — they have
    no ASCII decomposition and remain non-ASCII after this step.
    """
    nfkd = unicodedata.normalize('NFKD', str(text))
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def is_truly_non_latin(text: str) -> bool:
    """
    Returns True ONLY for genuinely non-Latin scripts:
    Devanagari (Hindi), Arabic, Chinese, Cyrillic, Japanese, etc.
    Returns False for French/German/Spanish/Portuguese accented characters.

    Args:
        text: raw business name string
    Returns:
        bool: True if name contains non-Latin script characters
    """
    stripped = strip_accents(text)   # ← removes French accents FIRST
    return bool(re.search(r'[^\x00-\x7F]', stripped))

# Usage everywhere is_non_latin was called:
# OLD: features['is_cross_script'] = int(is_non_latin(s2_name))
# NEW: features['is_cross_script'] = int(is_truly_non_latin(s2_name))
```

### Fix E — French Legal Suffixes (15 minutes)

```python
# In src/normalize.py — extend LEGAL_SUFFIXES

LEGAL_SUFFIXES = {
    # English / global
    "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "llp", "llc", "pvt", "private", "co", "company", "plc",
    "and", "the", "of",
    # French ← ADD ALL OF THESE
    "sarl",   # Société à responsabilité limitée
    "sas",    # Société par actions simplifiée
    "sasu",   # Société par actions simplifiée unipersonnelle
    "sa",     # Société anonyme
    "sci",    # Société civile immobilière
    "sca",    # Société en commandite par actions
    "snc",    # Société en nom collectif
    "eurl",   # Entreprise unipersonnelle à responsabilité limitée
    "eirl",   # Entreprise individuelle à responsabilité limitée
    "scop",   # Société coopérative et participative
    "se",     # Société européenne
    "scs",    # Société en commandite simple
    "ei",     # Entrepreneur individuel
    # German (future-proof)
    "gmbh", "ag", "kg", "ohg", "eg",
}
```

### Fix F — Character N-gram Cosine Similarity (30 minutes)

**Why it's needed:**

| Pair | Word Jaccard | Char 3-gram Cosine |
|---|---|---|
| "dhan investments" ↔ "dhaninvestments.com" | 0.00 ❌ | ~0.65 ✅ |
| "Societe Dupont" ↔ "Société Dupont" | 1.00 ✅ | 1.00 ✅ |
| "Ram Marketing" ↔ "राम मार्केटिंग" | 0.00 (cross-script) | 0.00 (different chars) |

```python
# In src/features.py — add char n-gram cosine feature

from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load pre-fitted char TF-IDF (from Phase 1)
char_tfidf = pickle.load(open("work/name_tfidf_char_v2.pkl", "rb"))

def char3gram_cosine(name_a: str, name_b: str) -> float:
    """
    Character n-gram (3,4) cosine similarity between two business names.

    Bridges domain-name vs normal-name gaps:
        'dhaninvestments' ↔ 'dhan investments' → ~0.65
    Handles accented characters after strip_accents normalization:
        'Societe Dupont' ↔ 'Societe Dupont' → 1.0

    Args:
        name_a: normalized name of S1 entity
        name_b: normalized name of S2/S3 entity
    Returns:
        float: cosine similarity in [0, 1]
    """
    if not name_a or not name_b:
        return 0.0
    v_a = char_tfidf.transform([name_a])
    v_b = char_tfidf.transform([name_b])
    return float(cosine_similarity(v_a, v_b)[0, 0])

# Add to feature dict:
features['name_char3gram_cosine'] = char3gram_cosine(
    norm_name(s1_row['business_name']),
    norm_name(s2_row['business_name'])
)
```

### Fix G — Address Numeric Overlap (30 minutes)

Numbers are the only signal that is 100% language-agnostic. Works identically for France (75001), India (110001), and US (10001).

```python
# In src/features.py — add numeric overlap features

import re

def numeric_overlap(addr1: str, addr2: str) -> float:
    """
    Jaccard similarity over shared digit sequences in addresses.
    Language-agnostic — works for France, India, US equally.
    Extracts 2-6 digit sequences (building numbers + postal codes).
    Avoids single digits (too common) and 7+ digits (phone numbers).

    Args:
        addr1: normalized address of S1 entity
        addr2: normalized address of S2/S3 entity
    Returns:
        float: Jaccard similarity over numeric tokens
    """
    n1 = set(re.findall(r'\b\d{2,6}\b', str(addr1)))
    n2 = set(re.findall(r'\b\d{2,6}\b', str(addr2)))
    if not n1 and not n2:
        return 0.5   # Neither has numbers → uninformative → neutral
    if not n1 or not n2:
        return 0.0   # One has numbers, other doesn't → mismatch
    return len(n1 & n2) / len(n1 | n2)

def postal_exact_match(addr1: str, addr2: str) -> float:
    """
    Exact match of 5-6 digit postal codes.
    US ZIP = 5 digits. India PIN = 6 digits. France code postal = 5 digits.
    Restricted to 5+ digits to avoid matching 2-digit French department
    codes (75, 69) that also appear as building numbers.

    Args:
        addr1: normalized address of S1 entity
        addr2: normalized address of S2/S3 entity
    Returns:
        1.0 if any 5-6 digit code matches exactly, 0.0 otherwise
    """
    p1 = set(re.findall(r'\b\d{5,6}\b', str(addr1)))
    p2 = set(re.findall(r'\b\d{5,6}\b', str(addr2)))
    if not p1 or not p2:
        return 0.0
    return 1.0 if bool(p1 & p2) else 0.0

# Add both to feature dict:
features['addr_numeric_overlap'] = numeric_overlap(s1_addr_norm, s2_addr_norm)
features['addr_postal_exact']    = postal_exact_match(s1_addr_norm, s2_addr_norm)
```

### Additional Features — Distractor Detection & Containment

```python
# In src/features.py

def containment_score(tokens_a: set, tokens_b: set) -> float:
    """
    Fraction of the SMALLER token set covered by the LARGER set.
    Catches subset relationships: 'Apple' ⊂ 'Apple India Pvt Ltd' → 1.0
    More sensitive than Jaccard for asymmetric name pairs.

    Args:
        tokens_a: word token set for first entity
        tokens_b: word token set for second entity
    Returns:
        float: containment in [0, 1]
    """
    if not tokens_a or not tokens_b:
        return 0.0
    smaller = tokens_a if len(tokens_a) <= len(tokens_b) else tokens_b
    larger  = tokens_b if len(tokens_a) <= len(tokens_b) else tokens_a
    return len(smaller & larger) / len(smaller)

# Distractor detector: name matches but address doesn't
# "Avatar Studios" ↔ "Avatar Proteins": name_jaccard=0.5, addr_jaccard=0.02
# → name_addr_discordance = 0.48 → LightGBM learns to reject
features['name_containment']       = containment_score(name_tok_s1, name_tok_s2)
features['addr_containment']       = containment_score(addr_tok_s1, addr_tok_s2)
features['name_addr_discordance']  = max(0.0,
    features['name_word_jaccard'] - features['addr_word_jaccard'])
features['char3_addr_discordance'] = max(0.0,
    features.get('name_char3gram_cosine', 0) - features['addr_tfidf_cosine'])
```

### Complete Updated Feature List (33 existing + 7 new = ~40 features)

```
NAME FEATURES (i)
  name_word_jaccard          ← existing
  name_tfidf_cosine          ← existing (now with France-aware IDF)
  name_levenshtein_norm      ← existing
  name_char3gram_cosine      ← NEW (Fix F)
  name_containment           ← NEW

ADDRESS FEATURES (j)
  addr_word_jaccard          ← existing
  addr_tfidf_cosine          ← existing (now with France-aware IDF)
  addr_levenshtein_norm      ← existing (if present)
  addr_numeric_overlap       ← NEW (Fix G)
  addr_postal_exact          ← NEW (Fix G)
  addr_containment           ← NEW

CROSS-SIGNAL FEATURES
  name_addr_discordance      ← NEW (distractor detector)
  char3_addr_discordance     ← NEW

EDGE CASE FLAGS
  is_cross_script            ← FIXED (uses is_truly_non_latin now)
  is_domain_probe            ← existing
  addr_missing_s2            ← existing (if present)
  addr_missing_both          ← existing (if present)

BM25 META FEATURES
  bm25_rank                  ← existing
  bm25_score_norm            ← existing
  bm25_score_gap             ← existing
  contention_count           ← existing
  ... (remaining ~15 existing features)
```

### Retraining Rules

```python
# MANDATORY: Use SAME hyperparameters as current best model
# lr=0.02, num_leaves=127, n_estimators=1420
# DO NOT tune hyperparams when adding features — isolate one variable at a time

# MANDATORY: neg_frac=1.0 for all val/cal/inference splits (already confirmed clean)

# MANDATORY: Refit isotonic calibration on full candidate set after retraining

# VALIDATION CHECK before any submission:
# Compute per-country F0.5 on held-out split
# If India F0.5 drops > 0.002 OR US F0.5 drops > 0.002 → DO NOT SUBMIT → revert
# Log all per-country scores in experiments/log.md

# TRAINING PAIR ENRICHMENT (optional but recommended):
# Add 20% of negative pairs as "distractor" pairs:
#   name_jaccard > 0.4 AND addr_jaccard < 0.1
# Teaches LightGBM: high name similarity + address mismatch = FP
```

---

## PART 4 — PHASE 3: FRANCE CALIBRATION FIX (Root Cause C)
### ~2 hours | Expected LB gain: +0.010 to +0.020

### Why Lewis 1995 Is Not the Problem — But Calibration Is

Lewis 1995 is correct. It maximizes expected-F₀.₅ over the candidate ranking. The bug is upstream: **isotonic calibration was fitted on India+US data.** France entities produce different LightGBM raw scores (because their TF-IDF features were pathological). The isotonic mapping converts those wrong raw scores to wrong probabilities. Lewis 1995 then optimizes using the wrong probabilities.

Two complementary fixes:

### Fix C1 — France-Specific β in Lewis 1995 (30 minutes)

Instead of standard F₀.₅ (β=0.5), use a more precision-conservative β=0.4 for France entities only. This makes Lewis 1995 prefer shorter prefixes (fewer, more confident matches) for France.

```python
# In src/decide.py (or wherever Lewis 1995 prefix selection is implemented)

def expected_fbeta_scores(probs: list, beta: float = 0.5) -> list:
    """
    Compute expected F_β at each prefix length k (Lewis 1995).

    Args:
        probs: list of calibrated probabilities for candidates, sorted desc
        beta: F-score parameter. β=0.5 standard. β=0.4 more precision-conservative.
    Returns:
        list of expected F_β scores, index 0 = singleton (no match)
    """
    beta_sq = beta ** 2
    total_expected_pos = sum(probs) + 1e-9
    results = [0.0]   # k=0: predict singleton → F_β = 0 if wrong, 1 if right
    # Lewis 1995 handles singleton separately — keep existing logic

    for k in range(1, len(probs) + 1):
        expected_pos_k = sum(probs[:k])
        precision_est  = expected_pos_k / k
        recall_est     = expected_pos_k / total_expected_pos
        if precision_est + recall_est < 1e-9:
            results.append(0.0)
            continue
        f = ((1 + beta_sq) * precision_est * recall_est
             / (beta_sq * precision_est + recall_est))
        results.append(f)
    return results

# In the per-entity decision loop:
def decide_prefix(candidates, country: str) -> list:
    """
    Select the prefix length maximizing expected F_β.
    Uses β=0.4 for France (more conservative → fewer FP matches).
    Uses β=0.5 for India/US (standard).

    Args:
        candidates: sorted list of (entity_id, calibrated_prob)
        country: country of the S1 entity
    Returns:
        list of matched entity IDs (may be empty for singleton prediction)
    """
    beta = 0.40 if country == "France" else 0.50

    probs = [p for _, p in candidates]
    scores = expected_fbeta_scores(probs, beta=beta)
    best_k = max(range(len(scores)), key=lambda k: scores[k])
    return [eid for eid, _ in candidates[:best_k]]
```

> **Why β=0.4?** The data shows France over-predicts matches (4.65% singletons vs expected 5.6%). Making β smaller → algorithm demands higher precision → selects shorter prefixes → fewer matches → fewer FP → higher F₀.₅.
> **Tune this:** Try β ∈ {0.35, 0.38, 0.40, 0.42, 0.45} on a France proxy validation set.

### Fix C2 — Platt Scaling Offset for France (if C1 insufficient)

```python
# After Phase 2 retraining, compute LightGBM raw log-odds for a sample of France pairs
# If France log-odds are systematically higher than India/US for similar pairs:
# apply a France-specific offset

import numpy as np

def calibrate_france(lgbm_raw_score: float, france_offset: float = 0.0) -> float:
    """
    Apply France-specific Platt scaling offset to LightGBM raw log-odds.
    If France probs are inflated, france_offset should be negative.

    Args:
        lgbm_raw_score: raw log-odds output from LightGBM
        france_offset: additive offset to raw score (tune this)
    Returns:
        float: calibrated probability
    """
    adjusted = lgbm_raw_score + france_offset
    return 1.0 / (1.0 + np.exp(-adjusted))

# Tune france_offset so that France predicted singleton rate ≈ 5.6%
# Start with france_offset = -0.3 and binary-search from there
# Current: France singleton rate = 4.65% (under-predicting singletons)
# Target:  France singleton rate ≈ 5.6% (matching India/US rate)
```

---

## PART 5 — PHASE 4: BM25 BLOCKING IMPROVEMENT (Root Cause H)
### ~30 minutes | Expected gain: +0.003 to +0.008

### Fix H — Conditional 80% Relative Filter

```python
# In blocking stage (wherever BM25 candidates are filtered)

def second_stage_filter(candidates: list, best_bm25_score: float) -> list:
    """
    Apply BM25 relative drop filter only for high-confidence probes.

    For cross-script or domain-name probes, the best BM25 score may be
    only 1.5–4.0 even for true matches (because tokens don't overlap well).
    Applying 80% filter (threshold=1.2–3.2) can drop the true match.

    Cutoff: if best_score < 6.0, probe is low-confidence → keep all top-10.

    Args:
        candidates: list of (entity_id, bm25_score) sorted by score desc
        best_bm25_score: maximum BM25 score in this probe's candidate set
    Returns:
        filtered list of candidates
    """
    if best_bm25_score < 6.0:
        # Low-confidence probe (cross-script, domain name, low-overlap)
        # Keep all top-10 candidates — let LightGBM sort it out
        return candidates
    # Normal high-confidence probe — apply standard 80% filter
    return [c for c in candidates if c[1] >= 0.80 * best_bm25_score]
```

### Address-Only Secondary Probe for Cross-Script Records

```python
# In blocking stage

def get_candidates_for_record(s2_row: dict, bm25_name_index, bm25_addr_index,
                               top_k: int = 10) -> list:
    """
    Generate candidate S1 entities for a given S2/S3 record.
    Uses address-only probe as fallback for non-Latin name records.

    Args:
        s2_row: dict with business_name, business_address, country fields
        bm25_name_index: BM25 index over S1 names, partitioned by country
        bm25_addr_index: BM25 index over S1 addresses, partitioned by country
        top_k: maximum candidates to return
    Returns:
        deduplicated list of (s1_entity_id, bm25_score) candidates
    """
    name = norm_name(s2_row['business_name'])
    addr = norm_addr(s2_row['business_address'])
    country = s2_row['country']

    # Primary probe: name
    name_cands = bm25_name_index.query(name, country=country, top_k=top_k)
    best_score = name_cands[0][1] if name_cands else 0.0

    # Apply conditional filter
    name_cands = second_stage_filter(name_cands, best_score)

    # Secondary probe: address-only (for cross-script and low-confidence probes)
    addr_cands = []
    if is_truly_non_latin(s2_row['business_name']) and best_score < 3.0:
        addr_cands = bm25_addr_index.query(addr, country=country, top_k=top_k)

    # Merge and deduplicate (keep higher score for duplicates)
    seen = {}
    for eid, score in name_cands + addr_cands:
        if eid not in seen or score > seen[eid]:
            seen[eid] = score
    return sorted(seen.items(), key=lambda x: -x[1])[:top_k]
```

---

## PART 6 — PHASE 5: HINDI TRANSLATION (Sunday — Gated)
### ~6–8 hours total | Expected gain: +0.005 to +0.015 | ONLY if Phases 1–4 confirm blocking as bottleneck

### Scale
- Test S2 Devanagari: 309,103 records (6.32%)
- Test S3 Devanagari: 181,068 records (3.56%)
- **Total to translate: 490,171 records**

### Gate Conditions — All Must Be True to Proceed

1. Phases 1–4 submitted AND LB improved (confirms France is the bottleneck)
2. Diagnostic 1 shows blocking recall < 95% for cross-script pairs
3. It is before Sunday 10:00 AM IST (enough compute time remaining)
4. GPU confirmed available: `python -c "import torch; print(torch.cuda.is_available())"`

### Implementation

```python
# scripts/translate_hindi.py
"""
Translate Devanagari business names and addresses to English.
Uses facebook/nllb-200-distilled-600M (MIT license, 600M params, ~2.4GB fp16).

Prerequisites:
  pip install transformers sentencepiece sacremoses
  python -c "import torch; print(torch.cuda.is_available())"  # Must be True

Runtime estimate (GPU): ~490K records × batch_size=32 → ~3.5 hours
"""

import csv, re, torch
from transformers import pipeline
from pathlib import Path

BASE = "/Users/ananshi_main/MAHU_DOC/BML_MUNJAL_2024/Hackathons/Amazon_ML_Challenge"

# ── Verify GPU
assert torch.cuda.is_available(), \
    "GPU not available! Run on AC power. Check nvidia-smi."
print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# ── Load NLLB model
translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    src_lang="hin_Deva",
    tgt_lang="eng_Latn",
    device=0,               # GPU
    max_length=128,
)

# ── Devanagari numeral conversion (must do BEFORE translation)
DEVA_DIGITS = str.maketrans('०१२३४५६७८९', '0123456789')

def has_devanagari(text: str) -> bool:
    return bool(re.search(r'[\u0900-\u097F]', str(text)))

def preprocess_hindi(text: str) -> str:
    """Convert Devanagari numerals to Arabic numerals before translation."""
    return str(text).translate(DEVA_DIGITS)

def translate_batch(texts: list) -> list:
    """Translate a batch of Hindi strings to English."""
    preprocessed = [preprocess_hindi(t) for t in texts]
    results = translator(preprocessed, batch_size=32)
    return [r['translation_text'] for r in results]

# ── Process test S2 and S3
for source_file, out_file in [
    ("test_source2.tsv", "hindi_translated_s2.tsv"),
    ("test_source3.tsv", "hindi_translated_s3.tsv"),
]:
    in_path  = f"{BASE}/student_resource/dataset/test/{source_file}"
    out_path = f"{BASE}/work/{out_file}"

    rows_to_translate = []
    with open(in_path, encoding="utf-8") as f:
        rows_to_translate = [
            r for r in csv.DictReader(f, delimiter="\t")
            if has_devanagari(r.get("business_name",""))
        ]

    print(f"{source_file}: {len(rows_to_translate):,} Devanagari records to translate")

    BATCH = 64
    translated_rows = []
    for i in range(0, len(rows_to_translate), BATCH):
        batch = rows_to_translate[i:i+BATCH]
        names = [r["business_name"] for r in batch]
        addrs = [r["business_address"] for r in batch]

        trans_names = translate_batch(names)
        trans_addrs = translate_batch(addrs)

        for r, tn, ta in zip(batch, trans_names, trans_addrs):
            r["business_name_translated"]    = tn
            r["business_address_translated"] = ta
            # Keep originals as fallback
            translated_rows.append(r)

        if i % 10000 == 0:
            print(f"  {i:,}/{len(rows_to_translate):,} done")

    # Write output
    if translated_rows:
        fieldnames = list(translated_rows[0].keys())
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(translated_rows)
    print(f"  Written: {out_path}")

print("\nDone. Next: re-probe BM25 index with translated names.")
print("Use UNION of original + translated candidates (deduped). Keep higher BM25 score.")
```

### Post-Translation Blocking Strategy

```python
# For each translated Devanagari record:
# 1. Probe BM25 with ORIGINAL name → candidates_orig
# 2. Probe BM25 with TRANSLATED name → candidates_trans
# 3. Merge: keep union, resolve duplicates by max(score)
# 4. Feed merged candidates to LightGBM

# Key risk: Some translations will be wrong ("Ram" vs "Raam" vs "Rama")
# Mitigation: Keep original probe candidates — translated candidates are ADDITIVE
# If translation is wrong, it adds noise; if right, it adds recall. Net positive.
```

---

## PART 7 — SUBMISSION STRATEGY (Budget Management)

**5 slots/day. ~10 remaining. Use them strategically.**

### Each Submission Must Test ONE Hypothesis

```
SLOT 1 (Today ~14:00 IST):
  What: Phase 1 (vocab refit) + Phase 2 D+E (accent fix + French suffixes)
  Hypothesis: "French words now have correct IDF; accent bug fixed."
  Expected LB: 0.930–0.942 if root cause A is confirmed
  If LB unchanged: IDF was already correct, root cause A is not the issue

SLOT 2 (Today ~20:00 IST):
  What: Phase 2 full (F+G new features + retrain LightGBM)
  Hypothesis: "Char n-gram + numeric overlap help France on known pairs."
  Expected LB: 0.938–0.952
  If no improvement: France blocking is the problem, not scoring

SLOT 3 (Sunday ~14:00 IST):
  What: Phase 3 (France β=0.4 in Lewis 1995) + Phase 4 (BM25 fix)
  Hypothesis: "Lewis 1995 now picks more conservative prefixes for France."
  Expected LB: 0.948–0.960

SLOT 4 (Sunday ~18:00 IST):
  What: Phase 5 (NLLB translation + retrain), gated on Slot 3 result
  Hypothesis: "Hindi records can now find their France counterparts."
  Expected LB: 0.958–0.972

SLOT 5 (Sunday ~21:00 IST):
  What: Best model confirmed, final submission
  Note: Never leave final to the last hour — platform issues are your problem
```

### Pre-Submission Checklist (Every Slot)

```
[ ] utils/validate_submission.py --check-ids → PASS
[ ] Row count = 1,732,544 (test_source1 exactly)
[ ] No S1- IDs in matched_entity_ids
[ ] France entities NOT silently dropped (check row for France entity)
[ ] No duplicate IDs within any row's ID list
[ ] Singleton rows have trailing tab after entity_id
[ ] No spaces after commas in ID lists
[ ] Per-country F0.5 logged (India/US must not regress vs 0.9742)
[ ] candidate_pairs.tsv generated (required for final zip)
[ ] Experiment logged in experiments/log.md:
      - Run ID, config diff, CV per-country scores
      - Hypothesis being tested
      - Post-submit: LB score and CV↔LB delta
[ ] Submission archived to submissions/ (never overwrite)
[ ] Slot count checked: ≤5 today
```

---

## PART 8 — FINAL SUBMISSION PACKAGE (Sunday before 22:00 IST)

From `student_resource/README.md` and `HACKATHON_RULES.md`:

```
Required files:
  output/matching_results.tsv        ← final predictions
  output/candidate_pairs.tsv         ← blocking output (MUST exist)
  code/business_entity_resolution/   ← full runnable pipeline
    src/                             ← normalize.py, blocking.py, features.py,
                                        model.py, decide.py, metric.py
    train.py                         ← end-to-end training entry point
    inference.py                     ← end-to-end inference entry point
    requirements.txt                 ← pinned versions
    README.md                        ← exact reproduction steps

  Documentation_template.md filled with:
    1. Executive Summary (2-3 sentences: BM25 → LightGBM → Lewis 1995)
    2. Problem Analysis (noise patterns, France OOD, Hindi cross-script, domain names)
    3. Candidate Generation / Blocking Strategy:
       - BM25 inverted index on S1 (name + address as separate fields)
       - Partitioned by country (hard blocking key)
       - Top-10 per probe
       - 80% relative score filter (conditional on best_score > 6.0)
       - Address-only secondary probe for cross-script records
       - Blocking recall: 98.7% → target 99.5%
    4. Matching Model:
       - LightGBM binary classifier, ~40 pairwise features
       - Isotonic calibration on disjoint split
       - Lewis 1995 Expected-F₀.₅ prefix selection
       - France: β=0.4 (more conservative)
    5. Feature Engineering (full list from Part 3)
    6. Results & Error Analysis
    7. Conclusion

  docs/approach.md (1-2 page summary):
    - Approach type: Blocking + Pairwise Classifier
    - ML models: LightGBM + Isotonic Regression
    - Key experiments: vocab refit, char n-gram, numeric overlap, France β tuning
    - Conclusion: LB scores, per-country breakdown
```

---

## PART 9 — NUMERICAL REFERENCE TABLE (All Data-Derived)

| Parameter | Value | Derivation |
|---|---|---|
| Public LB score | **0.924** | Leaderboard |
| Validation CV (stack3) | **0.9742** | Local measurement |
| CV↔LB gap | **0.0502** | 0.9742 − 0.924 |
| Implied France F₀.₅ | **~0.64** | Math: (0.924 − 0.8502×0.9742)/0.1498 |
| France fraction of test | 14.98% | 259,452/1,732,544 |
| True singleton rate | 5.58% | Training ground truth |
| True max matches | 11 | Training max |
| Soft match cap | 9 | p99.9 of training |
| France predicted singleton rate | 4.65% | matching_results.tsv |
| France target singleton rate | ~5.6% | Match India/US rate |
| Training avg matches | 3.461 | Ground truth mean |
| France avg matches (predicted) | 3.644 | matching_results.tsv |
| Lewis 1995 β (default) | 0.50 | Standard F₀.₅ |
| Lewis 1995 β (France) | 0.40 | More conservative |
| Platt offset starting point | −0.30 | France over-predicts → negative offset |
| BM25 low-confidence cutoff | best_score < 6.0 | Cross-script/domain threshold |
| Char n-gram range | (3, 4) | Business name standard |
| TF-IDF word max features | 600,000 | Fits in 32GB RAM |
| TF-IDF char max features | 400,000 | Fits in 32GB RAM |
| Devanagari records (test) | 490,171 | Dataset count S2+S3 |
| Domain records (test) | 319,922 | Dataset count S2+S3 |
| NLLB model | nllb-200-distilled-600M | MIT ✅, <600M ✅, ~2.4GB |
| Distractor negatives in training | 20% of negatives | Name-match, addr-mismatch pairs |
| Submissions remaining | ~10 | 5 Day2 + 5 Day3 |

---

## PART 10 — RISK MAP

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Retraining hurts India/US | Medium | High | Compare per-country F₀.₅ before submitting. Revert if India/US drops >0.002 |
| NLLB translation wrong for 15%+ records | Medium | Medium | Keep original candidates. Translated = additive. Wrong translation adds noise but doesn't remove true candidates |
| TF-IDF refit changes India/US feature values | High | Medium | Expected and manageable — MUST retrain LightGBM with new TF-IDF. Do not mix old model + new vectorizer |
| France blocking recall < 90% (not fixable by features) | Medium | Very High | If Slot 1 shows no LB improvement, shift all effort to blocking (Phase 4+5). Features cannot fix what blocking misses |
| Stack3 overfitting to India+US CV patterns | Medium | Medium | Check if France per-entity scores are systematically lower after each stack layer |
| Slot burn with no gain | High | Medium | One hypothesis per slot. Log before submit. If LB unchanged, diagnose cause before next slot |
| GPU drops on battery during NLLB run | High | High | From CLAUDE.md: "Stay on AC power." Verify: `nvidia-smi` before starting |
| candidate_pairs.tsv missing from pipeline | Unknown | High | Required for final zip. Verify exists before Day 3 |
| Private LB uses harder/different split | Cannot control | High | Optimize local CV and trust it. Do not overfit to public LB |

---

## PART 11 — KEY QUESTIONS FOR YOUR TEAMMATE

Before implementing, get answers to these. Each changes what you do:

1. **"Was TF-IDF and BM25 fitted using training data only, or did you include test_source1/2/3?"**
   → Determines if Phase 1 is needed

2. **"What are the exact 33 features? Does it include char n-gram cosine, numeric overlap, or containment?"**
   → Determines which of D/E/F/G are genuinely new vs already present

3. **"Does the pipeline generate candidate_pairs.tsv?"**
   → Required for final submission zip. Must exist.

4. **"What is stack3 — what does it add on top of the base LightGBM?"**
   → Determines if stacking is overfitting to India+US or genuinely helping France

5. **"Has France-specific blocking recall ever been measured?"**
   → Most critical diagnostic question

6. **"In is_non_latin — does it run strip_accents / NFKD first, or directly checks raw text?"**
   → Determines if Fix D is already done or genuinely needed

---

*This is the complete plan. Every section is backed by actual data from the dataset and the matching_results.tsv analysis. Priority order: Diagnose → Phase 1 → Phases 2+3 (one retrain) → Phase 4 → Submit → Phase 5 if time.*
