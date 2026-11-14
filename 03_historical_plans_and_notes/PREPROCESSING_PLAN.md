# 🧹 Preprocessing Master Plan
## Amazon ML Challenge 2026 — Business Entity Resolution

> **2-Day Sprint Guide | For Beginners | Written by your AI Guide**

---

## 📌 What is Preprocessing and Why Does It Matter?

Think of it like this: You're comparing two messy handwritten name cards. Before you compare them, you wipe the smudges, correct obvious abbreviations, and standardize spellings. That's preprocessing.

Our data has:
- `"Orelee's Barbershop"` in Source 1
- `"Orelees barber shop"` in Source 2 (probably the same, but a computer sees them as 0% similar)

**Without preprocessing**, our ML model will miss obvious matches.  
**With good preprocessing**, we can turn a "0% similar" match into a "95% similar" match — automatically.

---

## 📊 What We Know From the Data (EDA Findings)

| Finding | What It Means For Us |
|---|---|
| **2.2M** Source 1 records, **5M+** in S2 & S3 | Cannot compare everything — need smart filtering |
| Matches **NEVER cross country borders** | Free blocking key: only compare within same country |
| **48%** of S1 records share the same business name | Name alone is not enough — address is the tiebreaker |
| **168,967** empty addresses in Source 2 | Must handle missing addresses carefully |
| **4%** of S2/S3 records have **website domains** as names | e.g., "dhaninvestments.com" instead of "Dhan Investments" |
| Source 2 has **Hindi text** (Devanagari script) | Can match S1 via address even when name is 0% similar |
| Null/empty values written as "null", "n/a", "-" | Must convert these to blank **before** comparing |
| France appears **only in test**, never in training | Our code must be country-agnostic |

---

## 🗂️ Step-by-Step Preprocessing Plan

### STEP 0: Setup & Loading (Don't Load Everything at Once!)
**The problem:** The dataset is 2.35 GB. Loading it all at once on a laptop will crash Python.  
**The solution:** We process in chunks using Pandas `chunksize`.

```python
import pandas as pd

# NEVER do this for the full file:
# df = pd.read_csv("train_source2.tsv", sep="\t")  <- WILL CRASH

# DO this instead (process in chunks of 100,000 rows at a time):
for chunk in pd.read_csv("train_source2.tsv", sep="\t", chunksize=100_000, 
                          dtype=str, keep_default_na=False):
    # process chunk here
    pass
```

**Key flags to always use:**
- `sep="\t"` — the files are tab-separated, not comma-separated
- `dtype=str` — load everything as text, no accidental type conversions
- `keep_default_na=False` — prevent pandas from silently converting "NA", "None", etc. into NaN

---

### STEP 1: Fix Fake "Null" Values (Most Critical!)
**The problem:** Missing addresses aren't blank — they are written as "null", "n/a", "-", etc. If we leave them, our model might think two different businesses both located at "null" are the same!

**The action:**
```python
# Define all the "fake null" values
PLACEHOLDER = {"null", "n/a", "na", "nan", "nil", "none", "-", ""}

def clean_placeholder(text):
    """Convert fake null values to actual empty string."""
    if str(text).strip().lower() in PLACEHOLDER:
        return ""
    return text
```

**Apply to:** `business_name` and `business_address` in ALL source files (S1, S2, S3).

---

### STEP 2: Handle Non-Latin Scripts (Unicode Normalization)
**The problem:** Source 2 has names like Hindi text. Source 1 has "Ram Marketing Private Limited". A computer sees them as 0% similar.

**The insight from EDA:** Even when names are in different scripts, the **addresses** are usually in English/numbers. So we can use address matching to catch these. But we can also improve by stripping accents for French text (test set).

**The action:**
```python
import unicodedata

def normalize_unicode(text):
    """Remove accents from French/European characters.
    e.g., 'Cafe' gets normalized, accents removed
    """
    # NFKD decomposes characters into base + accent parts
    normalized = unicodedata.normalize("NFKD", str(text))
    # Remove combining characters (the accent marks)
    return "".join(c for c in normalized if not unicodedata.combining(c))
```

**Note:** For Hindi/Devanagari text, we do NOT try to translate — we just leave it as-is and rely on address matching to link those pairs.

---

### STEP 3: Strip Corporate Suffixes
**The problem:** "Inc", "Corp", "Pvt", "Ltd", "LLC" etc. appear differently across sources. "Acme Inc" vs "Acme Corporation" are the same company.

**The action:** Remove the most common legal suffixes from business names.

```python
# All common legal suffix forms
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation",
    "ltd", "limited", "llp", "llc",
    "pvt", "private", "co", "company",
    "sarl", "sas", "sa", "sci", "gmbh", "plc",
    "and", "the", "of"
}

def strip_legal_suffixes(text):
    """Remove legal suffixes from business names."""
    words = text.lower().split()
    filtered = [w for w in words if w not in LEGAL_SUFFIXES]
    return " ".join(filtered) if filtered else " ".join(words)
```

**Important:** We keep the original name too! We store BOTH the `cleaned_name` (without suffixes) AND the `original_name`. The ML model can use both.

---

### STEP 4: Core String Normalization
**The problem:** "McDonald's" vs "mcdonalds" vs "Mc Donalds" are all the same but look different.

**The action:** Apply these in order:
1. Normalize unicode (from Step 2)
2. Lowercase everything
3. Remove punctuation (replace with space)
4. Collapse multiple spaces into one
5. Strip leading/trailing spaces

```python
import re

def normalize_name(text):
    """Master cleaning function for business names."""
    if not text or str(text).strip() == "":
        return ""
    
    # Step 1: Normalize unicode (remove accents)
    text = normalize_unicode(str(text))
    
    # Step 2: Lowercase
    text = text.lower()
    
    # Step 3: Remove all non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Step 4: Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    # Step 5: Strip
    return text.strip()
```

---

### STEP 5: Handle Website Domains in Names
**The problem (from EDA):** ~4% of Source 2/3 records have a **website URL** as the business name.  
Example: "dhaninvestments.com" instead of "Dhan Investments".

**The action:**
```python
import re

# Pattern to detect website domains
DOMAIN_PATTERN = re.compile(
    r"\b[\w\-]+\.(com|net|org|in|fr|co|io|biz|info)\b", 
    re.IGNORECASE
)

def is_website_domain(text):
    """Return True if the name looks like a website domain."""
    return bool(DOMAIN_PATTERN.search(str(text)))

def extract_from_domain(text):
    """Try to extract business name from domain.
    e.g., 'dhaninvestments.com' -> 'dhaninvestments'
    """
    match = DOMAIN_PATTERN.search(str(text))
    if match:
        return text[:match.start() + match.end() - len(match.group(1)) - 1].strip()
    return text
```

---

### STEP 6: Address Abbreviation Expansion
**The problem:** "123 Main St, NY" vs "123 Main Street, New York" are the same address.

**The action:** Map common abbreviations to their full forms.

```python
ADDRESS_ABBREVS = {
    " rd ": " road ",
    " rd,": " road,",
    " st ": " street ",
    " st,": " street,",
    " ave ": " avenue ",
    " ave,": " avenue,",
    " blvd ": " boulevard ",
    " dr ": " drive ",
    " ln ": " lane ",
    " ct ": " court ",
    " hwy ": " highway ",
    " pkwy ": " parkway ",
    " marg ": " road ",   # India-specific
}

def expand_address_abbreviations(text):
    """Expand common address abbreviations."""
    text = " " + text.lower() + " "
    for abbrev, full in ADDRESS_ABBREVS.items():
        text = text.replace(abbrev, full)
    return text.strip()
```

**Warning:** Only expand abbreviations in addresses, NOT in business names. "St. Jude Hospital" should not have "St" expanded to "Street".

---

### STEP 7: Create Comparison-Ready Tokens
After all cleaning, we create **tokens** (individual words) for both name and address. These tokens are used later for similarity matching.

```python
def get_tokens(text, drop_legal=False):
    """Split cleaned text into a set of tokens."""
    words = text.lower().split()
    if drop_legal:
        return set(w for w in words if w not in LEGAL_SUFFIXES)
    return set(words)

def get_trigrams(text):
    """Break text into 3-character chunks for fuzzy matching.
    e.g., 'mcdonalds' -> {'mcd', 'cdo', 'don', 'ona', 'nal', 'ald', 'lds'}
    This is powerful because it catches typos!
    """
    clean = re.sub(r"[^a-z0-9]", "", text.lower())
    if len(clean) >= 3:
        return {clean[i:i+3] for i in range(len(clean)-2)}
    return {clean}
```

---

## 📋 What Each Cleaned Column Will Look Like

For every record, after preprocessing, we will have these new columns:

| New Column | Example | Purpose |
|---|---|---|
| `clean_name` | `"mcdonalds"` | Normalized business name |
| `clean_name_no_suffix` | `"mcdonalds"` (suffix removed) | For TF-IDF matching |
| `clean_address` | `"123 main street new york"` | Normalized address |
| `name_tokens` | `{"mcdonalds"}` | For Jaccard similarity |
| `name_trigrams` | `{"mcd", "cdo", ...}` | For fuzzy matching |
| `is_domain` | `False` | Flag for website-name records |
| `country` | `"US"` | Used as a hard blocking key |
| `has_address` | `True` | Flag for missing addresses |

---

## 🖥️ How to Set Up Anaconda So We Can Work Together

### Option A: I Write Scripts, You Run Them (RECOMMENDED for speed)
I am already connected to your file system. I can:
1. Write complete Python scripts (.py files) directly to your ML_Hackathon folder
2. Execute them via the terminal
3. Read the output and tell you what to do next

**You don't need to do anything special.** Just tell me: "Go ahead and write the preprocessing script" and I'll build it and run it for you.

### Option B: Jupyter Notebook in Anaconda
If you prefer working in a notebook:
1. Open **Anaconda Navigator**
2. Launch **JupyterLab**
3. Navigate to: `/Users/ananshi_main/MAHU_DOC/BML_MUNJAL_2024/ML_Hackathon`
4. I will create a `.ipynb` notebook file there that you can open and run cell-by-cell

### Required Libraries (Install Once)
Open Anaconda Prompt (or terminal) and run:
```bash
pip install pandas numpy scikit-learn lightgbm rapidfuzz
```

---

## 🗓️ 2-Day Sprint Timeline

### TODAY (Hours 1-4)
- [ ] Install required libraries
- [ ] Run preprocessing script on a **sample** (50,000 rows) to verify it works
- [ ] Check the output — do the cleaned names look right?

### TODAY (Hours 4-8)
- [ ] Run preprocessing on the **full dataset** (save cleaned files)
- [ ] Begin blocking strategy (TF-IDF nearest neighbors by country)

### TOMORROW (Hours 1-6)
- [ ] Build feature engineering (similarity scores for candidate pairs)
- [ ] Train XGBoost/LightGBM model on labeled training pairs
- [ ] Generate `matching_results.tsv` and `candidate_pairs.tsv`

### TOMORROW (Hours 6-8)
- [ ] Validate output with the provided `validate_submission.py` script
- [ ] Submit to the portal!

---

## Complete Preprocessing Order

```
Raw Data
    |
    v
1. Fix fake nulls ("null", "n/a", "-" -> "")
    |
    v
2. Remove unicode accents (for French text)
    |
    v
3. Lowercase everything
    |
    v
4. Remove punctuation (replace with space)
    |
    v
5. Collapse multiple spaces
    |
    v
6. Detect and flag website domain names
    |
    v
7. Strip legal suffixes from names (separate "core name" column)
    |
    v
8. Expand address abbreviations (St->Street, Rd->Road)
    |
    v
9. Generate tokens and trigrams
    |
    v
Clean Data (ready for blocking + ML)
```

---

> **KEY INSIGHT from EDA:** A **rare-token index** (using words that appear infrequently across all businesses) can achieve **99.78% blocking recall**. Name-only blocking caps at 85%. We will build this in the Blocking phase.
