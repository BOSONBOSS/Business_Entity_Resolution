"""
Fast Benchmark Dataset Construction Engine (PyArrow-Powered)
------------------------------------------------------------
Constructs exactly 50,000 balanced, stratified pairs:
- 25,000 True Matches (y=1) across US & India, including non-Latin script, domain names, zero-pad addresses.
- 25,000 Hard Negatives (y=0):
    * Type A (30% / 7,500): Name collision (same normalized name, different business/address)
    * Type B (30% / 7,500): Geographic collision (same city/state, different business name)
    * Type C (40% / 10,000): In-country random non-matches

Output: data/processed/benchmark_pairs.csv
"""

import os
import re
import time
import unicodedata
import pandas as pd
import numpy as np

TRAIN_DIR = "dataset/dataset/train"
OUTPUT_DIR = "data/processed"
BENCHMARK_PATH = os.path.join(OUTPUT_DIR, "benchmark_pairs.csv")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def is_non_latin(text):
    if not isinstance(text, str):
        return False
    return any(unicodedata.category(c) == 'Lo' and ord(c) > 0x024F for c in text)

def is_domain_or_handle(text):
    if not isinstance(text, str):
        return False
    t = text.lower().strip()
    return t.startswith('@') or any(t.endswith(ext) for ext in ['.com', '.in', '.org', '.net', '.co'])

def normalize_simple_name(text):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\b(pvt|ltd|inc|llc|corp|limited|private|services|retail|enterprises|sarl|sas)\b', '', t)
    return " ".join(t.split())

def extract_city(addr):
    if not isinstance(addr, str) or not addr.strip():
        return ""
    parts = [p.strip() for p in addr.split(',') if p.strip()]
    if len(parts) >= 2:
        return parts[-2].lower()
    return parts[-1].lower() if parts else ""

def build_benchmark():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 75)
    print("🚀 FAST BENCHMARK SAMPLING ENGINE (PyArrow Vectorized)")
    print("=" * 75)

    # 1. Load Ground Truth
    print("\n[1/5] Loading Ground Truth (train_ground_truth.tsv)...")
    gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t", engine="pyarrow")
    print(f"      Total S1 ground truth entries: {len(gt):,}")

    # Explode ground truth to get all positive pairs
    gt_valid = gt[gt['matched_entity_ids'].notna() & (gt['matched_entity_ids'] != '')].copy()
    print(f"      Non-singleton S1 entities: {len(gt_valid):,}")

    # Sample ~12,000 S1 entities to keep memory lightweight while having ~45,000 candidate match links
    gt_sample = gt_valid.sample(n=12000, random_state=RANDOM_SEED).copy()
    
    # Fast explosion of comma-separated IDs
    gt_sample['matched_entity_ids'] = gt_sample['matched_entity_ids'].str.split(',')
    exploded_gt = gt_sample.explode('matched_entity_ids')
    exploded_gt['matched_entity_ids'] = exploded_gt['matched_entity_ids'].str.strip()
    exploded_gt = exploded_gt[exploded_gt['matched_entity_ids'] != '']
    print(f"      Exploded positive pair candidates: {len(exploded_gt):,}")

    needed_s1 = set(exploded_gt['source1_entity_id'])
    needed_probes = set(exploded_gt['matched_entity_ids'])
    needed_s2 = {x for x in needed_probes if x.startswith('S2-')}
    needed_s3 = {x for x in needed_probes if x.startswith('S3-')}
    print(f"      Target IDs needed: S1={len(needed_s1):,}, S2={len(needed_s2):,}, S3={len(needed_s3):,}")

    # 2. Load Source 1
    print("\n[2/5] Loading Source 1 (train_source1.tsv)...")
    s1_all = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", engine="pyarrow")
    
    # Filter for matched S1 records + sample an extra pool for hard negative generation
    s1_matched = s1_all[s1_all['entity_id'].isin(needed_s1)].copy()
    s1_extra = s1_all[~s1_all['entity_id'].isin(needed_s1)].sample(n=25000, random_state=RANDOM_SEED)
    s1_pool = pd.concat([s1_matched, s1_extra], ignore_index=True)
    print(f"      S1 working pool: {len(s1_pool):,} records (matched: {len(s1_matched):,})")
    del s1_all

    # 3. Load Source 2 & Source 3
    print("\n[3/5] Loading Source 2 & Source 3 probes...")
    s2_all = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", engine="pyarrow")
    s2_matched = s2_all[s2_all['entity_id'].isin(needed_s2)].copy()
    s2_extra = s2_all[~s2_all['entity_id'].isin(needed_s2)].sample(n=25000, random_state=RANDOM_SEED)
    s2_pool = pd.concat([s2_matched, s2_extra], ignore_index=True)
    s2_pool['probe_source'] = 'S2'
    print(f"      S2 working pool: {len(s2_pool):,} records")
    del s2_all

    s3_all = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", engine="pyarrow")
    s3_matched = s3_all[s3_all['entity_id'].isin(needed_s3)].copy()
    s3_extra = s3_all[~s3_all['entity_id'].isin(needed_s3)].sample(n=25000, random_state=RANDOM_SEED)
    s3_pool = pd.concat([s3_matched, s3_extra], ignore_index=True)
    s3_pool['probe_source'] = 'S3'
    print(f"      S3 working pool: {len(s3_pool):,} records")
    del s3_all

    probes_pool = pd.concat([s2_pool, s3_pool], ignore_index=True).drop_duplicates(subset=['entity_id'])
    print(f"      Total unique probes in pool: {len(probes_pool):,}")

    # Compute normalization features before dictionary conversion
    s1_pool['norm_name'] = s1_pool['business_name'].apply(normalize_simple_name)
    s1_pool['city'] = s1_pool['business_address'].apply(extract_city)
    probes_pool['norm_name'] = probes_pool['business_name'].apply(normalize_simple_name)
    probes_pool['city'] = probes_pool['business_address'].apply(extract_city)

    # Build fast dictionary lookups
    s1_dict = s1_pool.set_index('entity_id').to_dict('index')
    probe_dict = probes_pool.set_index('entity_id').to_dict('index')

    # 4. Construct Positive Pairs
    print("\n[4/5] Constructing & Stratifying 25,000 Positive Pairs...")
    gt_match_set = set(zip(exploded_gt['source1_entity_id'], exploded_gt['matched_entity_ids']))
    
    pos_list = []
    for s1_id, p_id in gt_match_set:
        if s1_id in s1_dict and p_id in probe_dict:
            s1_info = s1_dict[s1_id]
            p_info = probe_dict[p_id]
            
            p_name = str(p_info['business_name']) if pd.notna(p_info['business_name']) else ""
            is_hindi = is_non_latin(p_name)
            is_dom = is_domain_or_handle(p_name)

            if is_hindi:
                ptype = "pos_non_latin"
            elif is_dom:
                ptype = "pos_domain_or_handle"
            else:
                ptype = "pos_standard"

            pos_list.append({
                's1_id': s1_id,
                's1_name': s1_info['business_name'],
                's1_address': s1_info['business_address'],
                's1_country': s1_info['country'],
                'probe_id': p_id,
                'probe_name': p_info['business_name'],
                'probe_address': p_info['business_address'],
                'probe_country': p_info['country'],
                'probe_source': p_info['probe_source'],
                'pair_type': ptype,
                'label': 1
            })

    pos_df = pd.DataFrame(pos_list)
    print(f"      Valid raw positive pairs assembled: {len(pos_df):,}")
    print("      Positive pair categories:\n", pos_df['pair_type'].value_counts())

    # Ensure all special strata (non-Latin, domain) are preserved, then fill remainder with country balance
    special_pos = pos_df[pos_df['pair_type'].isin(['pos_non_latin', 'pos_domain_or_handle'])]
    standard_pos = pos_df[pos_df['pair_type'] == 'pos_standard']
    
    needed_std = 25000 - len(special_pos)
    if needed_std > 0 and len(standard_pos) >= needed_std:
        sampled_std = standard_pos.sample(n=needed_std, random_state=RANDOM_SEED)
        final_pos = pd.concat([special_pos, sampled_std], ignore_index=True)
    else:
        final_pos = pos_df.sample(n=25000, random_state=RANDOM_SEED)

    print(f"      Selected exactly: {len(final_pos):,} Positive Pairs.")

    # 5. Construct 25,000 Hard Negatives
    print("\n[5/5] Generating 25,000 Hard Negatives (Type A, B, C)...")

    # Re-index for negative generation
    name_to_probes = probes_pool[probes_pool['norm_name'].str.len() > 3].groupby('norm_name')['entity_id'].apply(list).to_dict()
    city_to_probes = probes_pool[probes_pool['city'].str.len() > 3].groupby('city')['entity_id'].apply(list).to_dict()

    neg_list = []

    # Type A: Same normalized name, different address / different entity (7,500 pairs)
    print("      -> Generating Type A: Name Collisions (7,500 pairs)...")
    count_a = 0
    for _, s1_r in s1_pool.sample(frac=1.0, random_state=RANDOM_SEED).iterrows():
        nn = s1_r['norm_name']
        if nn in name_to_probes:
            for cand_id in name_to_probes[nn]:
                if (s1_r['entity_id'], cand_id) not in gt_match_set:
                    cand_p = probe_dict[cand_id]
                    neg_list.append({
                        's1_id': s1_r['entity_id'],
                        's1_name': s1_r['business_name'],
                        's1_address': s1_r['business_address'],
                        's1_country': s1_r['country'],
                        'probe_id': cand_id,
                        'probe_name': cand_p['business_name'],
                        'probe_address': cand_p['business_address'],
                        'probe_country': cand_p['country'],
                        'probe_source': cand_p['probe_source'],
                        'pair_type': 'neg_hard_name_collision',
                        'label': 0
                    })
                    count_a += 1
                    break
        if count_a >= 7500:
            break
    print(f"         Generated {count_a:,} Type A negatives.")

    # Type B: Geographic Collision (same city/state, different business) (7,500 pairs)
    print("      -> Generating Type B: Geographic Collisions (7,500 pairs)...")
    count_b = 0
    for _, s1_r in s1_pool.sample(frac=1.0, random_state=RANDOM_SEED + 1).iterrows():
        city = s1_r['city']
        if city in city_to_probes:
            candidates = city_to_probes[city]
            for cand_id in np.random.choice(candidates, size=min(10, len(candidates)), replace=False):
                cand_p = probe_dict[cand_id]
                if (s1_r['entity_id'], cand_id) not in gt_match_set and cand_p['norm_name'] != s1_r['norm_name']:
                    neg_list.append({
                        's1_id': s1_r['entity_id'],
                        's1_name': s1_r['business_name'],
                        's1_address': s1_r['business_address'],
                        's1_country': s1_r['country'],
                        'probe_id': cand_id,
                        'probe_name': cand_p['business_name'],
                        'probe_address': cand_p['business_address'],
                        'probe_country': cand_p['country'],
                        'probe_source': cand_p['probe_source'],
                        'pair_type': 'neg_hard_geographic_collision',
                        'label': 0
                    })
                    count_b += 1
                    break
        if count_b >= 7500:
            break
    print(f"         Generated {count_b:,} Type B negatives.")

    # Type C: Random in-country non-matches (remaining needed to reach 25,000)
    needed_c = 25000 - (count_a + count_b)
    print(f"      -> Generating Type C: Random In-Country Non-Matches ({needed_c:,} pairs)...")
    
    probes_us = probes_pool[probes_pool['country'] == 'US']['entity_id'].values
    probes_in = probes_pool[probes_pool['country'] == 'India']['entity_id'].values

    count_c = 0
    s1_shuffled = s1_pool.sample(frac=1.0, random_state=RANDOM_SEED + 2).reset_index(drop=True)
    idx = 0
    while count_c < needed_c and idx < len(s1_shuffled):
        s1_r = s1_shuffled.iloc[idx]
        idx += 1
        country = s1_r['country']
        pool = probes_us if country == 'US' else probes_in
        if len(pool) == 0:
            continue
        cand_id = np.random.choice(pool)
        if (s1_r['entity_id'], cand_id) not in gt_match_set:
            cand_p = probe_dict[cand_id]
            neg_list.append({
                's1_id': s1_r['entity_id'],
                's1_name': s1_r['business_name'],
                's1_address': s1_r['business_address'],
                's1_country': s1_r['country'],
                'probe_id': cand_id,
                'probe_name': cand_p['business_name'],
                'probe_address': cand_p['business_address'],
                'probe_country': cand_p['country'],
                'probe_source': cand_p['probe_source'],
                'pair_type': 'neg_random_incountry',
                'label': 0
            })
            count_c += 1

    print(f"         Generated {count_c:,} Type C negatives.")
    neg_df = pd.DataFrame(neg_list)

    # 6. Final Concatenation & Save
    benchmark_df = pd.concat([final_pos, neg_df], ignore_index=True)
    benchmark_df = benchmark_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    benchmark_df.to_csv(BENCHMARK_PATH, index=False)
    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(BENCHMARK_PATH) / (1024 * 1024)

    print("\n" + "=" * 75)
    print(f"✅ BENCHMARK GENERATION COMPLETE in {elapsed:.2f} seconds!")
    print(f"   Saved to: {BENCHMARK_PATH}")
    print(f"   Size: {file_size_mb:.2f} MB | Rows: {len(benchmark_df):,}")
    print("\n   Label Distribution:")
    for lbl, count in benchmark_df['label'].value_counts().items():
        print(f"     Label {lbl}: {count:,} ({count/len(benchmark_df)*100:.1f}%)")
    print("\n   Pair Difficulty Distribution:")
    for pt, count in benchmark_df['pair_type'].value_counts().items():
        print(f"     {pt:32s}: {count:,} ({count/len(benchmark_df)*100:.1f}%)")
    print("=" * 75)

if __name__ == "__main__":
    build_benchmark()
