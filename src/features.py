"""
Pairwise Feature Engineering Module
-----------------------------------
Transforms normalized business record pairs into a 15-dimensional numerical
feature vector x in R^15:
  1. name_token_jaccard       (Word token Jaccard)
  2. name_jaro_winkler        (RapidFuzz prefix-weighted string similarity)
  3. name_levenshtein_ratio   (RapidFuzz normalized edit distance)
  4. name_char_trigram_cosine (Sub-word 3-gram cosine similarity)
  5. name_len_diff_ratio      (Normalized length difference)
  6. addr_token_jaccard       (Address word token Jaccard)
  7. addr_jaro_winkler        (Address Jaro-Winkler ratio)
  8. addr_levenshtein_ratio   (Address normalized edit distance)
  9. addr_char_trigram_cosine (Address 3-gram cosine similarity)
 10. numeric_overlap_jaccard  (Jaccard similarity of extracted street/ZIP digits)
 11. zip_code_match           (Exact match of 5-digit US or 6-digit India postal codes)
 12. is_script_mismatch       (Boolean flag: Latin vs. non-Latin Indic script)
 13. both_addresses_missing   (Boolean flag: both records lack address)
 14. one_address_missing      (Boolean flag: asymmetric missing address)
 15. same_country             (Boolean flag: matching country of registration)

Output: data/processed/feature_matrix.csv
"""

import os
import sys
import re
import math
import time
from collections import Counter
import pandas as pd
import numpy as np
from rapidfuzz.distance import JaroWinkler, Levenshtein

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import (
    preprocess_business_name,
    preprocess_address,
    is_non_latin
)

FEATURE_COLUMNS = [
    'name_token_jaccard',
    'name_jaro_winkler',
    'name_levenshtein_ratio',
    'name_char_trigram_cosine',
    'name_len_diff_ratio',
    'addr_token_jaccard',
    'addr_jaro_winkler',
    'addr_levenshtein_ratio',
    'addr_char_trigram_cosine',
    'numeric_overlap_jaccard',
    'zip_code_match',
    'is_script_mismatch',
    'both_addresses_missing',
    'one_address_missing',
    'same_country'
]

US_ZIP_REGEX = re.compile(r'\b\d{5}\b')
IN_PIN_REGEX = re.compile(r'\b\d{6}\b')
DIGITS_REGEX = re.compile(r'\b\d+\b')


def token_jaccard(tokens1, tokens2) -> float:
    """Compute Jaccard similarity between two token lists/sets."""
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def char_trigram_cosine(s1: str, s2: str) -> float:
    """Compute character 3-gram cosine similarity."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if len(s1) < 3 or len(s2) < 3:
        return 1.0 if (s1 in s2 or s2 in s1) else 0.0
    
    t1 = [s1[i:i+3] for i in range(len(s1)-2)]
    t2 = [s2[i:i+3] for i in range(len(s2)-2)]
    c1, c2 = Counter(t1), Counter(t2)
    
    common = sum((c1 & c2).values())
    norm = math.sqrt(sum(v*v for v in c1.values()) * sum(v*v for v in c2.values()))
    return common / norm if norm > 0 else 0.0


def extract_postal_code(text: str):
    """Extract US 5-digit ZIP or Indian 6-digit PIN code."""
    if not text:
        return None
    pin = IN_PIN_REGEX.findall(text)
    if pin:
        return pin[-1]
    zip_code = US_ZIP_REGEX.findall(text)
    if zip_code:
        return zip_code[-1]
    return None


def extract_features_from_row(row):
    """Compute 15 features for a single paired record."""
    # Raw strings
    raw_s1_name = str(row['s1_name']) if pd.notna(row['s1_name']) else ""
    raw_p_name = str(row['probe_name']) if pd.notna(row['probe_name']) else ""
    raw_s1_addr = str(row['s1_address']) if pd.notna(row['s1_address']) else ""
    raw_p_addr = str(row['probe_address']) if pd.notna(row['probe_address']) else ""
    s1_country = str(row['s1_country']).strip() if pd.notna(row['s1_country']) else ""
    p_country = str(row['probe_country']).strip() if pd.notna(row['probe_country']) else ""

    # Cleaned strings
    c_s1_name = preprocess_business_name(raw_s1_name)
    c_p_name = preprocess_business_name(raw_p_name)
    c_s1_addr = preprocess_address(raw_s1_addr)
    c_p_addr = preprocess_address(raw_p_addr)

    # 1. Name Features
    s1_name_tokens = c_s1_name.split()
    p_name_tokens = c_p_name.split()

    f_name_jaccard = token_jaccard(s1_name_tokens, p_name_tokens)
    f_name_jw = JaroWinkler.similarity(c_s1_name, c_p_name)
    f_name_lev = Levenshtein.normalized_similarity(c_s1_name, c_p_name)
    f_name_trigram = char_trigram_cosine(c_s1_name, c_p_name)
    
    max_len = max(len(c_s1_name), len(c_p_name), 1)
    f_name_len_diff = abs(len(c_s1_name) - len(c_p_name)) / max_len

    # 2. Address Features
    s1_addr_tokens = c_s1_addr.split()
    p_addr_tokens = c_p_addr.split()

    addr1_empty = (len(s1_addr_tokens) == 0)
    addr2_empty = (len(p_addr_tokens) == 0)

    if addr1_empty or addr2_empty:
        f_addr_jaccard = 0.0
        f_addr_jw = 0.0
        f_addr_lev = 0.0
        f_addr_trigram = 0.0
        f_num_overlap = 0.0
        f_zip_match = 0.0
    else:
        f_addr_jaccard = token_jaccard(s1_addr_tokens, p_addr_tokens)
        f_addr_jw = JaroWinkler.similarity(c_s1_addr, c_p_addr)
        f_addr_lev = Levenshtein.normalized_similarity(c_s1_addr, c_p_addr)
        f_addr_trigram = char_trigram_cosine(c_s1_addr, c_p_addr)

        # Numeric tokens overlap (building numbers, street numbers)
        nums1 = DIGITS_REGEX.findall(c_s1_addr)
        nums2 = DIGITS_REGEX.findall(c_p_addr)
        f_num_overlap = token_jaccard(nums1, nums2)

        # Postal code exact match
        zip1 = extract_postal_code(raw_s1_addr)
        zip2 = extract_postal_code(raw_p_addr)
        f_zip_match = 1.0 if (zip1 and zip2 and zip1 == zip2) else 0.0

    # 3. Flags and Quality Indicators
    s1_is_nl = is_non_latin(raw_s1_name)
    p_is_nl = is_non_latin(raw_p_name)
    f_script_mismatch = 1.0 if (s1_is_nl != p_is_nl) else 0.0

    f_both_addr_missing = 1.0 if (addr1_empty and addr2_empty) else 0.0
    f_one_addr_missing = 1.0 if (addr1_empty != addr2_empty) else 0.0
    f_same_country = 1.0 if (s1_country and p_country and s1_country == p_country) else 0.0

    return [
        f_name_jaccard,
        f_name_jw,
        f_name_lev,
        f_name_trigram,
        f_name_len_diff,
        f_addr_jaccard,
        f_addr_jw,
        f_addr_lev,
        f_addr_trigram,
        f_num_overlap,
        f_zip_match,
        f_script_mismatch,
        f_both_addr_missing,
        f_one_addr_missing,
        f_same_country
    ]


def build_feature_matrix(input_csv="data/processed/benchmark_pairs.csv",
                         output_csv="data/processed/feature_matrix.csv"):
    start_time = time.time()
    print("=" * 75)
    print("🚀 EXTRACTING 15 PAIRWISE FEATURES FROM BENCHMARK DATASET")
    print("=" * 75)

    print(f"Loading input pairs from: {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Total rows to process: {len(df):,}")

    records = df.to_dict('records')
    features = []

    report_interval = 10000
    for idx, r in enumerate(records):
        features.append(extract_features_from_row(r))
        if (idx + 1) % report_interval == 0:
            print(f"  Processed {idx + 1:,} / {len(records):,} pairs...")

    feat_df = pd.DataFrame(features, columns=FEATURE_COLUMNS)

    # Attach labels and identifiers
    feat_df['s1_id'] = df['s1_id']
    feat_df['probe_id'] = df['probe_id']
    feat_df['pair_type'] = df['pair_type']
    feat_df['label'] = df['label']

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    feat_df.to_csv(output_csv, index=False)

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_csv) / (1024 * 1024)

    print("=" * 75)
    print(f"✅ FEATURE EXTRACTION COMPLETE in {elapsed:.2f} seconds!")
    print(f"   Saved to: {output_csv}")
    print(f"   Size: {file_size_mb:.2f} MB | Shape: {feat_df.shape}")
    print("\n   Feature Statistics Summary:")
    print(feat_df[FEATURE_COLUMNS].describe().T[['mean', 'std', 'min', '50%', 'max']])
    print("=" * 75)
    return feat_df


if __name__ == "__main__":
    build_feature_matrix()
