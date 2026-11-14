# 🧠 Master Optimization Plan v2 — Amazon ML Challenge 2026
## Restructured from Critical Evaluation | LB=0.924 → Target: 0.960+
### Correct Baseline: LB=0.924 | CV=0.9742 | Gap=0.0502 | Stack3 just finished

---

## 📌 Framing the Real Problem First

The previous plan was wrong in its framing. Here is the corrected picture.

```
CV  = 0.9742  (India+US validation, stack3)
LB  = 0.924   (public leaderboard, full test including France)
Gap = 0.0502  ← THIS is the problem. Not features. Not thresholds.
```

### Why the Gap Exists (Math)

France = 14.98% of test. Solving for France F₀.₅:

```
0.8502 × 0.9742  +  0.1498 × France_F₀.₅  =  0.924
                                France_F₀.₅  =  0.639
```

**France is scoring ~0.64 on the leaderboard. India+US score ~0.97.**  
A 0.33-point gap. This is not a threshold problem. This is a structural failure.

### Three Root Causes (From Critical Evaluation)

| # | Root Cause | Estimated LB Impact | Fixable? |
|---|---|---|---|
| 🔴 A | BM25 + TF-IDF IDF weights built on India+US only → French words OOV or wrong weight | ~0.025 | Yes — refit on full corpus |
| 🔴 B | LightGBM feature distributions on France are pathological → Lewis 1995 mis-selects prefix | ~0.020 | Partial — new features + retrain |
| 🔴 C | Isotonic calibration fitted on India+US → France probabilities systematically off | ~0.015 | Hard — needs France-like cal data |
| 🟠 D | Accent bug: French é/à/ç flagged as "Hindi" → is_non_latin wrong for 24% of France pairs | ~0.005 | Yes — 1 line fix |
| 🟠 E | French legal suffixes not stripped (SARL, EURL, etc.) | ~0.003 | Yes — dict addition |
| 🟠 F | Domain names: word Jaccard = 0.0 → no char n-gram fallback | ~0.004 | Yes — add feature |
| 🟡 G | Address numeric overlap missing (language-agnostic) | ~0.003 | Yes — add feature |

**Correct priority: Fix A first. Then B+D+E+F+G together (one retrain). Then C.**

---

## 🛑 CORRECTIONS TO THE PREVIOUS PLAN

These ideas from the previous plan were **wrong** and are removed:

1. **"France probability multiplier × 0.88 before Lewis 1995"** → WRONG.
   Lewis 1995 maximizes expected-F₀.₅ over the *ranked* candidate list. Uniform scaling preserves ranking. Same ranking → same prefix → same output. The multiplier does nothing.

2. **"Logistic Regression pre-filter"** → REMOVED.
   LR can't model name×address interaction. Risk of silently dropping true France pairs that LightGBM would have accepted. Compute is not the bottleneck.

3. **"Early removal of high-confidence matches"** → REMOVED.
   Risk of cascading errors. "Duplicate claim removal at end" is already the correct approach.

4. **"All expected gains" based on 0.954** → RECALIBRATED.
   Everything was calculated from the wrong baseline. Recalculated from 0.924 below.

---

## PHASE 0: DIAGNOSE BEFORE YOU FIX (30 minutes — do this FIRST)

**Before writing any feature code, answer these two questions. They determine what you do next.**

### Diagnostic 1: France Blocking Recall

If France blocking recall is below 95%, feature engineering cannot help — the ceiling is the recall. You'd be polishing features for pairs the model never sees.

```python
# Run on train data to estimate France-equivalent blocking recall
# Proxy: measure BM25 recall for India entities whose names use 
# words rare in training (simulates "OOD vocabulary" of France)

# More direct: look at current matching_results.tsv
# For France entities with 0 matches in submission (singletons):
# - Are these true singletons, or are they unmatchable due to blocking failure?
# Estimate: sample 200 France S1 entities, manually check if their S2/S3 
# counterparts have token overlap with S1 name/address
```

### Diagnostic 2: BM25 Score Distribution for France

```python
# Proxy via candidate_pairs.tsv (if generated) or from current matching pass
# Compare distribution of max(bm25_score) per probe:
# - India/US: expected range 5-20 for true matches
# - France: if mostly 1-3, French words are OOV in BM25 index → Cause A confirmed
```

**If France blocking recall < 95%: Fix A (vocabulary) is your entire Day 2.**
**If France blocking recall ≥ 95%: Fix B+D+E+F+G (features + retrain) is Day 2.**

---

## PHASE 1: THE VOCABULARY FIX (Root Cause A)
### Time: 2-3 hours | Expected LB gain: +0.020 to +0.030 | No labels needed

This is the highest-leverage fix and requires NO retraining labels or ground truth.

### What Went Wrong

The BM25 inverted index and TF-IDF vectorizers were built (fitted) on training data: `train_source1/2/3.tsv`. These files contain only India and US entities.

- French word "Société" → document frequency in training = 0 → BM25 IDF = log(N/0) = undefined or max
- French word "Rue" → not in training vocabulary → TF-IDF vectorizer maps it to OOV (0 weight)
- French word "Boulangerie" → same issue
- Result: BM25 score between a French S1 and a French S2 with identical names = near-zero, because neither record's tokens exist in the IDF table

### The Fix

```python
# Step 1: Collect ALL text (train + test, no labels used)
all_names = []
all_addresses = []

for filepath in [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv", 
    "dataset/train/train_source3.tsv",
    "dataset/test/test_source1.tsv",       # <-- CRITICAL: adds French vocab
    "dataset/test/test_source2.tsv",
    "dataset/test/test_source3.tsv",
]:
    with open(filepath, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            all_names.append(norm(row["business_name"]))
            all_addresses.append(norm(row["business_address"]))

# Step 2: Refit TF-IDF on combined corpus
from sklearn.feature_extraction.text import TfidfVectorizer

name_tfidf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    max_features=500_000,
    sublinear_tf=True,
)
name_tfidf.fit(all_names)   # French words now have correct IDF

addr_tfidf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    max_features=500_000,
    sublinear_tf=True,
)
addr_tfidf.fit(all_addresses)

# Step 3: Rebuild BM25 index using IDF weights from combined corpus
# (Your teammate's BM25 implementation — ask them to pass external_idf=combined_idf)
# OR: use the combined TF-IDF cosine as an alternative to BM25 for the blocking stage

# Step 4: Save vectorizers — these become new pipeline artefacts
import pickle
pickle.dump(name_tfidf, open("work/name_tfidf_v2.pkl", "wb"))
pickle.dump(addr_tfidf, open("work/addr_tfidf_v2.pkl", "wb"))
```

> ⚠️ **Critical:** If you refit TF-IDF, you MUST also retrain LightGBM with the new feature values. Old model + new TF-IDF → incorrect feature interpretations → do not mix.

### Why This Doesn't Leak Labels

Fitting a vectorizer only requires observing words — not labels. Using test-set vocabulary in the vectorizer is standard practice and does not leak any ground truth. You are only learning "what words exist in French", not "which French records match."

---

## PHASE 2: FEATURE FIXES (Root Cause B + D + E + F + G)
### Time: 4-6 hours including retrain | Expected LB gain: +0.015 to +0.025

Do all of these together in one retrain. Each fix is small; combined they are significant.

### Fix D: French Accent Bug (10 minutes — highest priority, lowest effort)

```python
import unicodedata, re

def strip_accents(text):
    """
    Remove diacritical marks so 'Société' becomes 'Societe'.
    Must be called BEFORE is_non_latin check — French accents are not Hindi.
    """
    nfkd = unicodedata.normalize('NFKD', str(text))
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def is_truly_non_latin(text):
    """
    Returns True only for Devanagari, Arabic, Chinese, Cyrillic etc.
    Returns False for French/German/Spanish accented characters.
    """
    stripped = strip_accents(text)               # Remove accents first!
    return bool(re.search(r'[^\x00-\x7F]', stripped))
```

**Impact:** Without this, 24.4% of France matched pairs incorrectly trigger the "Hindi" logic path. LightGBM applies address-reliance weights that were learned for Hindi→English cross-script pairs, not French→French same-script pairs.

### Fix E: French Legal Suffixes (15 minutes)

```python
# Add to existing LEGAL_SUFFIXES set in normalize.py
LEGAL_SUFFIXES = {
    # English / global
    "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "llp", "llc", "pvt", "private", "co", "company", "plc", "and", "the", "of",
    # French  ← ADD THESE
    "sarl", "sas", "sa", "sci", "sca", "snc", "eurl", "eirl",
    "scop", "sasu", "se", "scs",
    # German (for future countries)
    "gmbh", "ag", "eg", "kg", "ohg",
    # Indian
    "pvt", "ltd",
}
```

**Impact:** "Dépannage Express SARL" vs "Depannage Express" → after suffix strip: identical. Without this, Jaccard = 0.67 instead of 1.0. LightGBM scores it lower than it should.

### Fix F: Character N-gram Feature for Domain Names and Accented Names (30 minutes + retrain)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Fit on combined corpus (from Phase 1)
char_tfidf = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 4),
    max_features=300_000,
    sublinear_tf=True,
)
char_tfidf.fit(all_names)

# During feature computation for each pair:
def char3gram_cosine(name_a, name_b):
    """
    Character n-gram cosine similarity.
    Bridges: 'dhaninvestments.com' ↔ 'dhan investments' (~65% overlap)
    Bridges: 'Societe' ↔ 'Société' (after accent strip, near-perfect overlap)
    Fails gracefully on cross-script pairs (Devanagari ↔ Latin = 0).
    """
    v_a = char_tfidf.transform([norm(name_a)])
    v_b = char_tfidf.transform([norm(name_b)])
    return float((v_a * v_b.T).toarray()[0, 0])

# Add as LightGBM feature:
features['name_char3gram_cosine'] = char3gram_cosine(s1_name, s2_name)
```

### Fix G: Address Numeric Overlap — The Language-Agnostic Tiebreaker (30 minutes)

This feature works identically for France, Hindi, and English because numbers are universal.

```python
import re

def numeric_overlap(addr1, addr2):
    """
    Jaccard over digit sequences (building nos., postal codes).
    Language-agnostic — works for France, India, US equally.
    
    Args:
        addr1: normalized address string for S1 entity
        addr2: normalized address string for S2/S3 entity
    Returns:
        float: Jaccard similarity over shared numeric tokens
    """
    # Extract 2-6 digit sequences (catches building numbers + postal codes)
    # Avoid single digits (1, 2) — too common; avoid 7+ digits (phone numbers)
    n1 = set(re.findall(r'\b\d{2,6}\b', str(addr1)))
    n2 = set(re.findall(r'\b\d{2,6}\b', str(addr2)))
    if not n1 and not n2:
        return 0.5   # Neither has numbers → uninformative, neutral
    if not n1 or not n2:
        return 0.0   # One has numbers, other doesn't → mismatch signal
    return len(n1 & n2) / len(n1 | n2)

def postal_exact_match(addr1, addr2):
    """
    Exact match of 5-6 digit codes (US ZIP, India PIN, France code postal).
    Very high precision signal — if postal matches, entities are co-located.
    
    Note: avoid matching 2-digit French department codes (75, 69) as these
    appear as building numbers too — restrict to 5+ digits.
    """
    p1 = set(re.findall(r'\b\d{5,6}\b', str(addr1)))
    p2 = set(re.findall(r'\b\d{5,6}\b', str(addr2)))
    if not p1 or not p2:
        return 0.0
    return 1.0 if bool(p1 & p2) else 0.0

# Add both as LightGBM features:
features['addr_numeric_overlap']  = numeric_overlap(s1_addr, s2_addr)
features['addr_postal_exact']     = postal_exact_match(s1_addr, s2_addr)
```

### Additional Features — Distractor Handling (Avatar Studios vs Avatar Proteins)

```python
# Detects name match with address mismatch — the core distractor signature
features['name_addr_discordance'] = max(0.0,
    features['name_word_jaccard'] - features['addr_word_jaccard'])
# High value = name similar, address very different → almost certainly wrong match

# Token containment (catches "Apple" ⊂ "Apple India Pvt Ltd")
def containment(tokens_a, tokens_b):
    if not tokens_a or not tokens_b:
        return 0.0
    smaller = tokens_a if len(tokens_a) <= len(tokens_b) else tokens_b
    larger  = tokens_b if len(tokens_a) <= len(tokens_b) else tokens_a
    return len(smaller & larger) / len(smaller)

features['name_containment'] = containment(name_tokens_s1, name_tokens_s2)
features['addr_containment'] = containment(addr_tokens_s1, addr_tokens_s2)
```

### The Retrain

After all fixes D, E, F, G are in place:

```python
# IMPORTANT: Use same hyperparams as current best model
# lr=0.02, num_leaves=127, n_estimators=1420 (from teammate's methodology)
# DO NOT change hyperparams when adding features — isolate the variable

# Retrain with new features on same training fold
# Refit isotonic calibration on same calibration split (full neg_frac=1.0)

# Before submitting: compare per-country F0.5 on validation
# If India/US drops even 0.002 → revert to old model
# Track in experiments/log.md: feature diff + CV per-country + LB score
```

---

## PHASE 3: CALIBRATION FIX FOR FRANCE (Root Cause C)
### Time: 2-3 hours | Expected LB gain: +0.010 to +0.020 | Hardest fix

Lewis 1995 expected-F₀.₅ prefix selection is correct in mechanism. The problem is the inputs: **calibrated probabilities for France are not calibrated to real-world France match probability.**

Isotonic calibration was fitted on India+US data. France entities have different feature distributions → the isotonic mapping gives wrong probabilities → Lewis 1995 optimizes the wrong objective.

**How Lewis 1995 works (for context):**
```
For each S1 entity, sort candidates by prob (desc).
For prefix k (taking top-k candidates):
    E[F0.5(k)] = (1.25 × k_positives_expected × recall_expected) 
                  / (0.25 × k_positives_expected + recall_expected)
Pick the k that maximizes E[F0.5(k)], including k=0 (singleton).
```

If all France probs are wrong by a systematic factor, Lewis 1995 will pick the wrong k. For example: if France probs are all inflated by 1.5x, the model adds more candidates than it should → false positives → lower precision → lower F0.5.

### The Calibration Fix: France-Specific Isotonic Layer

**Option 1: Holdout-based (best if data exists)**
If any France entities exist in training (they don't — France is test-only), use them as France calibration data.
→ Not available for this dataset.

**Option 2: Proxy calibration using cross-language India holdout**
Hold out all India entities whose addresses are mostly non-English (using Hindi-language addresses as proxy for OOD data). Train isotonic calibration on India+US, then recalibrate on this "hard" India holdout. Apply the recalibration to France.
→ Approximate, but better than nothing.

**Option 3: Direct France calibration via test data characteristics**
From matching_results.tsv analysis: France predicts 3.64 avg matches (should be ~3.46 based on training). The model predicts 0.94% too many matches for France. 

Lewis 1995 can be made more conservative for France by changing β (the F-score parameter):
```python
# Standard F0.5: β=0.5 (precision weighted 4× over recall in Lewis formulation)
# For France, use β=0.4 (precision weighted even more heavily)
# This forces Lewis 1995 to accept shorter prefixes for the same probability estimates
# Implementation: pass beta=0.4 to the expected_fscore function for France entities only

def expected_fβ(probs, beta=0.5):
    """Expected F_β at each prefix length."""
    beta_sq = beta ** 2
    sorted_p = sorted(probs, reverse=True)
    results = [0.0]  # k=0, singleton
    for k in range(1, len(sorted_p) + 1):
        expected_pos = sum(sorted_p[:k])
        recall_estimate = expected_pos / (sum(sorted_p) + 1e-9)
        precision_estimate = expected_pos / k
        f = ((1 + beta_sq) * precision_estimate * recall_estimate 
             / (beta_sq * precision_estimate + recall_estimate + 1e-9))
        results.append(f)
    return results

# For France only:
if country == "France":
    scores = expected_fβ(candidate_probs, beta=0.4)  # More precision-conservative
else:
    scores = expected_fβ(candidate_probs, beta=0.5)  # Standard
```

> This is mathematically valid and doesn't break the decision rule — it just makes the algorithm more conservative for France, which we know has inflated probabilities from the singleton rate analysis.

**Option 4: Platt Scaling on France feature distribution**
After Phase 2 retraining, compute LightGBM raw scores (log-odds) for a sample of France entities. Check if the score distribution is shifted vs India/US. If France scores are uniformly 0.3 log-odds higher, apply a France-specific offset:
```python
if country == "France":
    prob = 1 / (1 + exp(-(lgbm_score - FRANCE_OFFSET)))  # Subtract offset
else:
    prob = 1 / (1 + exp(-lgbm_score))
```
Use FRANCE_OFFSET tuned on the France singleton rate target (should reach ~5.6%).

---

## PHASE 4: BM25 BLOCKING IMPROVEMENT
### Time: 1-2 hours | Expected gain: +0.005 to +0.010

### Fix the 80% Relative Filter for Low-Confidence Probes

```python
def second_stage_filter(candidates, best_bm25_score):
    """
    Apply BM25 relative filter only for high-confidence probes.
    
    For cross-script pairs (Hindi ↔ English) or domain names, the best BM25 
    score might be 1.5–3.0 even for true matches. Applying 80% filter 
    (threshold=1.2–2.4) can drop the true match.
    
    Empirical cutoff: if best_score < 6.0, the probe is "low confidence."
    Keep all top-10 candidates and let LightGBM decide.
    """
    if best_bm25_score < 6.0:
        return candidates            # Low confidence → keep all top-10
    return [c for c in candidates if c.bm25_score >= 0.80 * best_bm25_score]
```

### Address-Only Secondary Probe for Cross-Script Pairs

For Devanagari-name S2/S3 records where the BM25 name probe returns best_score < 3.0:
```python
if is_truly_non_latin(s2_name) and best_name_bm25 < 3.0:
    # Retry probe using address only (English address is more reliable)
    addr_candidates = bm25_address_index.query(s2_address, country=s2_country, top_k=10)
    # Merge with name candidates (union, deduplicated)
    candidates = deduplicate(name_candidates + addr_candidates)
```

---

## PHASE 5: HINDI TRANSLATION (Sunday Only — Conditional)
### Time: 6-8 hours | Expected gain: +0.005 to +0.015 | Do ONLY if Phase 1-2 succeed

**Gate condition:** Only implement if:
1. Phases 1-4 have been submitted and LB improves (confirms France is the bottleneck)
2. Blocking recall diagnostic shows <95% for cross-script pairs
3. It is before Sunday 10 AM IST (enough time to run + submit before deadline)

```python
# Model: facebook/nllb-200-distilled-600M
# License: MIT ✅ | Params: 600M ✅ | VRAM: ~2.4GB fp16 ✅ (fits in 8GB)
# Speed: ~30-40 records/sec on GPU → 490K records → ~3.5 hours

# Step 1: Separate Devanagari records
# Step 2: Translate names and addresses (keep originals as fallback)
# Step 3: Re-probe BM25 index with translated names
# Step 4: Merge translated + original candidate sets (union)
# Step 5: Retrain LightGBM (translated training data included)
# Step 6: Validate that India/US F0.5 did not regress
```

**Key risk:** Translation variance. "राम" → "Ram" or "Rama" or "Raam". Mitigate by using union of translated + original candidates.

---

## 📅 48-HOUR EXECUTION PLAN

### NOW → 6:00 AM IST (4 hours)
**Goal: Diagnose + Fix A + Fix D (no model change)**

| Time | Action | Output |
|---|---|---|
| 01:54–02:30 | Run blocking recall diagnostic (France BM25 score distribution) | Know if blocking is the problem |
| 02:30–04:30 | Refit TF-IDF + char TF-IDF on full train+test corpus | `work/name_tfidf_v2.pkl`, `work/addr_tfidf_v2.pkl` |
| 04:30–05:30 | Fix accent bug (`strip_accents` before `is_non_latin`) | One-line code fix in normalize.py |
| 05:30–06:00 | Add French legal suffixes to normalization dict | Dict update in normalize.py |

### 6:00 AM → 12:00 PM (6 hours)
**Goal: Retrain with all new features**

| Time | Action | Output |
|---|---|---|
| 06:00–07:00 | Add features: char3gram_cosine, numeric_overlap, postal_exact, containment, name_addr_discordance | New feature columns in build_pairs |
| 07:00–09:00 | Rebuild training pairs with new features (Phase 1 vectorizers) | `work/train_pairs_v2.parquet` |
| 09:00–11:00 | Retrain LightGBM + refit isotonic calibration | `work/lgbm_v2.pkl` |
| 11:00–12:00 | Validate: per-country F0.5 on holdout. India/US should not drop vs CV=0.9742 | Validation report |

### 12:00 PM → 14:00 PM (2 hours)
**Goal: Apply France calibration fix + submit**

| Time | Action | Output |
|---|---|---|
| 12:00–13:00 | Implement France β=0.4 or Platt offset. Run inference on test | `output/matching_results.tsv` |
| 13:00–13:30 | Run `utils/validate_submission.py --check-ids` → must be PASS | Validated submission |
| 13:30–14:00 | Submit (Slot 1 of Day 2). Log in experiments/log.md | LB score → expected ~0.940–0.950 |

### 14:00 PM → 20:00 PM (6 hours)
**Goal: Fix BM25 blocking + address-only secondary probe + submit**

| Time | Action | Output |
|---|---|---|
| 14:00–16:00 | Fix BM25 80% rule (conditional on best_score). Add address-only probe for cross-script | Blocking improvement |
| 16:00–18:00 | Regenerate candidates + rebuild pairs + retrain | `work/lgbm_v3.pkl` |
| 18:00–20:00 | Inference + validate + submit (Slot 2 of Day 2) | LB score → expected ~0.950–0.958 |

### Sunday: 09:00 AM → 20:00 PM
**Goal: Translation + final model**

| Time | Action | Output |
|---|---|---|
| 09:00–09:30 | Decision gate: did Day 2 submissions improve LB? Is blocking confirmed as bottleneck? | Go/No-go for NLLB |
| 09:30–13:00 | NLLB translation of 490K Devanagari records + address conversion | `work/hindi_translated.parquet` |
| 13:00–16:00 | Rebuild pairs + retrain LightGBM with translated data | `work/lgbm_v4.pkl` |
| 16:00–18:00 | Final inference + validate + submit | Best submission |
| 18:00–22:00 | Fill `Documentation_template.md`, write `docs/approach.md` (1-2 pages) | Required artefacts |
| 22:00–23:30 | Final submission if any slots remain. Archive all submissions. | Done |
| 23:59 | DEADLINE | — |

---

## 📊 Revised Expected Score Trajectory

| Phase | Change | Expected LB | Confidence |
|---|---|---|---|
| Baseline | Stack3, no change | 0.924 | Confirmed |
| Phase 1 + D + E | Vocabulary refit + accent fix + French suffixes | 0.938–0.948 | High |
| Phase 2 (F + G + retrain) | Char n-gram + numeric overlap + distractor features | 0.948–0.958 | High |
| Phase 3 | France calibration fix (β=0.4 or Platt offset) | 0.955–0.965 | Medium |
| Phase 4 | BM25 blocking improvement | 0.957–0.968 | Medium |
| Phase 5 (if blocking confirmed) | NLLB Hindi translation | 0.960–0.975 | Low-Medium |

---

## 🔑 Key Thresholds and Parameters (All Data-Derived)

| Parameter | Value | Source |
|---|---|---|
| True singleton rate (training) | 5.58% | Ground truth analysis |
| True max matches | 11 | Ground truth max |
| Soft match cap | 9 (p99.9) | Ground truth percentiles |
| France F0.5 (implied from LB) | ~0.64 | Math: LB=0.924, India+US≈0.97 |
| France β for Lewis 1995 | 0.4 (try) | More precision-conservative |
| BM25 low-confidence cutoff | best_score < 6.0 | Avoids cross-script over-filtering |
| France predicted singleton rate | 4.65% (current) | matching_results.tsv analysis |
| France target singleton rate | ~5.6% | Matches India/US observed rate |
| Devanagari records in test | 490,171 | Dataset count |
| Domain name records in test | 319,922 | Dataset count |
| Char n-gram range | (3, 4) | Standard for business names |
| TF-IDF max features (word) | 500,000 | Memory: fits in 32GB |
| TF-IDF max features (char) | 300,000 | Memory: fits in 32GB |
| NLLB model | facebook/nllb-200-distilled-600M | MIT ✅, <600M params ✅, ~2.4GB |
| Remaining submission slots | ~10 (5 Day2 + 5 Day3) | Budget management |

---

## ✅ Pre-Submission Checklist (Every Submission)

- [ ] `utils/validate_submission.py --check-ids` → **PASS**
- [ ] Row count = 1,732,544 (test_source1 row count exactly)
- [ ] France entities present — not silently dropped
- [ ] No S1- IDs in matched_entity_ids column
- [ ] No duplicate IDs within any row's ID list
- [ ] `candidate_pairs.tsv` also generated (required for final zip)
- [ ] Per-country F0.5 on validation logged (India/US should not regress)
- [ ] Experiment logged in `experiments/log.md` with: run id, config diff, CV per-country, hypothesis
- [ ] Submission slot count checked (≤5 today)
- [ ] Submission archived to `submissions/` — never overwrite

---

## 📄 Final Submission Artefacts Required

From `student_resource/README.md` and `HACKATHON_RULES.md`:

1. `output/matching_results.tsv` — final predictions
2. `output/candidate_pairs.tsv` — blocking output fed to LightGBM
3. `code/business_entity_resolution/` — full runnable pipeline
4. `requirements.txt` — pinned versions
5. `Documentation_template.md` — filled with:
   - Methodology used
   - Candidate generation / blocking strategy
   - Model architecture (LightGBM + isotonic calibration + Lewis 1995)
   - Feature engineering (all 40+ features)
6. `docs/approach.md` — 1-2 page summary (approach, models, experiments, conclusion)

---

*v2.0 — Restructured from critical evaluation. Priority is: Diagnose → Vocabulary Fix → Feature Retrain → Calibration → Submit. The CV↔LB gap is the only problem worth solving.*
