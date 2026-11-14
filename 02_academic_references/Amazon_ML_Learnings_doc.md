# **Business Entity Resolution** 

**Methodology and engineering record of the 0.9806 submission** 

**0.9806 0.985890 99.7964% 95.6569% Reported leaderboard Validation macro F0.5 Validation precision Validation recall** 

This document explains the submitted system, its development, its measured results and the limits of the evidence. It reconstructs the early Claude work and the later Codex work from imported conversations, source code, saved model metadata, experiment logs and final artifact manifests. It is a historical and technical record, not a claim that every experiment discussed was deployed. 

**Submission identity:** `pipeline/output_FINAL_20260927_2328/matching_results.tsv` . 

**Leaderboard: 0.9806** , reported by the user after uploading the recommended file. **Reporting validation:** macro F0.5 **0.985890249416** , pair precision **0.997963556140** , pair recall **0.956568593563** . The validation set was repeatedly inspected and contains US and India only. Its score is not an estimate with a demonstrated guarantee for unseen countries. 

**Final system:** country-partitioned lexical retrieval → 38-feature LightGBM → 19 additional listwise features and a second LightGBM → calibrated epoch-2 mDeBERTa blend → entity-level prefix selection and unique probe ownership → conservative XLM-R admissions and sparse-candidate rescues. 

The later `output_FINAL_20260927_2355` file is a separate experiment with no recovered leaderboard result. It is deliberately excluded from the definition of the 0.9806 model. 

_Prepared 28 September 2026_ 

_24 chapters · architecture and workflow diagrams · artifact evidence and reproduction guide_ 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **Contents** 

|1. What exactly was submitted|3|
|---|---|
|2. Problem, contracts and metric|5|
|3. Architecture and data flow|6|
|4. EDA: measurements that changed the design|7|
|5. Text normalization and learned vocabulary|10|
|6. Candidate generation and retrieval ablations|12|
|7. Exact LightGBM feature dictionary|14|
|8. Training splits, LightGBM fitting and calibration|17|
|9. Cross-encoder training and full-coverage scoring|20|
|10. Core probability blend, prefix selection and ownership|23|
|11. The final recall improvements|25|
|12. Final validation and leaderboard interpretation|27|
|13. Development chronology and accepted milestones|28|
|14. Failed, neutral and incomplete experiments|30|
|15. Hybrid retrieval handoff: promising, not shipped|33|
|16. Engineering workflow and resource placement|34|
|17. Validation design, failure modes and uncertainty|36|
|18. Lessons and corrections worth carrying forward|37|
|19. Reproducing the winning result|38|
|20. Generalization and the next research phases|40|
|21. Provenance, attribution and remaining gaps|42|
|22. Reconciliation of nearby artifacts|43|
|23. Artifact hashes and evidence index|44|
|24. Research references and terminology|47|



Historical record · 28 September 2026 

2 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **1. What exactly was submitted** 

The winning artifact is a complete prediction system rather than a single estimator. Four learned components contribute: the v5 pair classifier, the v5 listwise classifier, the epoch-2 mDeBERTa cross-encoder, and XLM-R used only for a conservative admission rule. Isotonic calibrators convert their scores to the probability scales expected by downstream decisions. No new training was necessary for the final recall improvement. 

The core submission before the final additions was `output_v5_ceFULL_20260927_0527` . Every assignment in that file was retained. The final process admitted extra matches only for previously unclaimed S2/S3 records, with separate validation of precision, recall and entity-macro F0.5. Preserving old assignments avoids stealing an incumbent match; it does **not** mathematically guarantee unchanged precision, because any new assignment can still be wrong. 

|**Artifact fact**|**Value**|
|---|---|
|Required S1 rows|1,732,544|
|Incumbent matched pairs|5,670,481|
|XLM-R additions|22,786|
|Sparse rescues|1,105|
|Final matched pairs|5,694,372|
|S1 entities changed by additions|23,759|
|Conflicting new owners vetoed on<br>test|0|
|Final exported candidate pairs|17,079,913|
|Official validation|Passed; zero errors and warnings|



The exact matching-file SHA256 is: 

```
132e6a2daf9ae75b86746d8f71af6f3db3c34ee445295a0cfa14628a88cf8ea4
```

The exact candidate-file SHA256 is: 

```
d0f465403b23998afd61562a9b9fb94484b225361766e76651d12e5b3ff44581
```

The final directory also contains `added_pairs.parquet` , `manifest.json` , `validation.json` and `official_validation.json` . These identify the source inputs and the acceptance evidence. The final writer was recorded at source commit `8eddc31` ; final acceptance validation at `73f97a0` ; the completed submission record was checkpointed at `f05433d` . [E01–E03] 

Historical record · 28 September 2026 

3 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

### **1.1 Included and excluded components** 

|**Component**|**Status in the 0.9806 submission**|
|---|---|
|BM25 top-10, name:address 1:1|Included|
|Core relative BM25 cutoff 0.8|Included; a narrow rescue branch considers 0.6–0.8|
|v5 stage 1: 38 features, 127 leaves|Included|
|v5 stage 2: 57 total features, 127<br>leaves|Included|
|mDeBERTa checkpoint after two<br>epochs|Included, full retained-candidate coverage|
|XLM-R|Included for high-confidence unclaimed-probe admissions|
|Equal average of two cross-encoders|Not the winning method|
|Epoch-3/4 mDeBERTa checkpoint|Not used|
|Dense/E5 fused candidate set|Not used; available handoff was train-only|
|BM25 b=0.3, alternative field<br>weights|Tested but not deployed|
|S1 name-ambiguity feature /<br>class-weight experiments|Not in the frozen v5 models|
|Country-specific gate,<br>prior-matching, hard state filter|Not used|
|Post-contention backfill / later v6<br>admissions|Not used in this artifact|



Historical record · 28 September 2026 

4 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **2. Problem, contracts and metric** 

Source 1 is the deduplicated reference table. For each S1 business, the required output is the set of S2 and S3 records representing that business. An S1 business can have zero, one or many matches. Each S2/S3 record has at most one correct S1 parent. This is **many probes to one reference** , not a one-record-per-source bipartite matching problem. 

The four source columns are `entity_id` , `business_name` , `business_address` and `country` . The source prefix in the entity ID separates S1, S2 and S3. The ground truth maps an S1 ID to a comma-separated S2/S3 ID list, with an empty list for an unmatched S1. A quote-aware TSV reader is required: fields may contain punctuation and quoting, so neither a comma parser nor an arbitrary binary tab splitter is a safe substitute. [E04, E05] 

The official output has two columns: `source1_entity_id` and `matched_entity_ids` . Every test S1 appears exactly once, even when unmatched. IDs inside each list are unique and valid. The companion `candidate_pairs.tsv` uses `candidate_entity_ids` ; only the matching file is leaderboard-scored. An early assumption that candidate count itself incurred a leaderboard penalty was incorrect. Candidate size still matters operationally, but it is not a direct leaderboard objective. [E04, E07] 

### **2.1 Why the objective is not accuracy or average precision** 

For a reference entity e, let T_e be its true match set and P_e the predicted set. For nonempty truth, with TP_e = |T_e intersect P_e|: 

```
F0.5(e) = 1.25 TP_e / (0.25 |T_e| + |P_e|)
         = 1.25 TP_e / (1.25 TP_e + 0.25 FN_e + FP_e)
Score    = mean of F0.5(e) over ALL S1 entities
```

If both sets are empty, the entity earns 1. If truth is empty and any match is emitted, it earns 0. If truth is nonempty and prediction is empty, it earns 0. An unmatched S1 is called a “singleton” in the historical logs; in this document that means zero S2/S3 matches, not an entity with exactly one matching probe. 

The formula gives a false positive four times the denominator weight of a false negative, for a fixed TP count. Historical shorthand saying “precision is weighted twice” is not the literal count-based formula. Macro averaging also makes each S1 equally important: an erroneous merge on an unmatched entity can be disproportionately costly compared with a missed member of a large correct set. 

Pair precision and recall are separately reported as TP/(TP+FP) and TP/(TP+FN), pooling pairs. They must not be substituted into a single F formula and labelled entity-macro F0.5. LightGBM average precision and transformer accuracy are training diagnostics; neither is the final selection metric. [E06] 

### **2.2 Retrieval direction and identity encoding** 

The pipeline indexes S1 and queries it with S2/S3. Each probe needs its single correct parent somewhere in top-k; querying in the opposite direction would need an S1 query to retrieve a variable-size set of true probes. This choice was supported by the data partition audit and inspired by Sparkly's top-k blocking design. This implementation uses Tantivy and local workers; it is not the Sparkly Spark/Lucene codebase. [R1, E08] 

Within the numerical pipeline, `s1_eid` is the numeric suffix of the S1 ID. A probe key is `numeric_id + source_number * 2^40` , keeping S2 and S3 namespaces disjoint. Numeric IDs are join/group keys, not LightGBM input features. `probe_src` , however, is one of the retained v5 inputs. 

Historical record · 28 September 2026 

5 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **3. Architecture and data flow** 



<!-- Start of picture text -->
Raw S1/S2/S3 TSVs<br>name - address - country<br>Country-partitioned lexical retrieval<br>separate-field BM25 - k=10 - boosts 1:1<br>@renieeS Cra Raw record-pair serialization<br>Baipalninputshrcountny! IDF mDeBERTa epoch 2 + isotonic<br>[PAT tin wy) full core coverage<br>Core candidates: bm25_rel >= 0.8 |<br>1<br>17,078,808 test pairs |<br>Stage 1 + isotonic -> pl |<br>38 inputs - 3,000 trees<br>19 listwise features + stage 2 + isotonic<br>57 inputs - 3,995 trees -> p2<br>p = 0.30 p2 + 0.70 calibrated CE<br>Prefix selector: beta=.5, gate=.6<br>Global one-parent ownership<br>Conservative XLM-R admissions<br>+ sparse rescues (.6 <= bm25_rel < .8)<br>Only unclaimed probes<br>Immutable final merge + validation<br>5,694,372 matches - 1,732,544 S1 rows<br>Reported leaderboard: 0.9806<br><!-- End of picture text -->

_Figure 1. Winning inference architecture. Solid branches contribute to the 0.9806 artifact; the two conservative additions are applied after the core owners have been fixed._ 

The design separates five questions. Retrieval asks which parents are plausible. Pair scoring asks whether the two records look like the same business. Listwise scoring asks how convincing that pair is relative to the other candidates for its S1. The cross-encoder reads the original two records jointly, supplying information lost by scalar similarity features. The final selector chooses a set, then ownership and conservative admissions enforce the output structure. 

The lexical feature branch uses normalized text. The transformer branch serializes raw name and address fields with column markers; it does not consume the 38 handcrafted features or a BM25 score. These two views are complementary. Country is used to partition lexical search and select IDF statistics but is absent from the transformer text. 

The final inference graph is asymmetric: mDeBERTa influences every retained core candidate through blending, whereas XLM-R can add only an exceptionally confident unclaimed match. A failed global averaging experiment therefore did not make the XLM-R checkpoint useless; the final system reused it through a more selective decision rule. 

Historical record · 28 September 2026 

6 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **4. EDA: measurements that changed the design** 

### **4.1 Complete source inventory** 

The final raw-data audit scanned all six source files, 24,229,173 records. Its results were independently reproduced with a second quote-aware parser. The initial scan took 44.94 seconds on the i9 with two streaming workers; the independent recount took 40.77 seconds. These were data audits, not model evaluations. All six source hashes were recorded. [E05] 

|**File**|**Rows**|**Country counts**|**Blank addresses**|
|---|---|---|---|
|train S1|2,206,821|US: 1,323,633, India: 883,188|0|
|train S2|5,034,616|India: 2,017,799, US: 3,016,817|168,967|
|train S3|5,285,603|US: 3,170,056, India: 2,115,547|175,916|
|test S1|1,732,544|US: 663,106, France: 259,452, India: 809,986|0|
|test S2|4,887,273|India: 2,312,565, France: 703,378, US: 1,871,330|129,408|
|test S3|5,082,316|India: 2,405,000, France: 731,615, US: 1,945,701|136,098|



There are 2,206,821 labelled reference entities and 7,638,365 true training pairs. The S2/S3 training pool contains 10,320,219 records, so many probes have no labelled parent. Mean true matches per S1 is approximately 3.46; mean available probes per S1 is 4.68 in train and 5.75 in test. These are different denominators and must not be conflated. 

|**True matches per S1**|**S1 entities**|**Share**|
|---|---|---|
|0|123,247|5.5848%|
|1|119,157|5.3995%|
|2|375,212|17.0024%|
|3|530,841|24.0546%|
|4|484,115|21.9372%|
|5|321,957|14.5892%|
|6|164,868|7.4708%|
|7|63,968|2.8986%|
|8|18,680|0.8465%|
|9|4,205|0.1905%|
|10|534|0.0242%|
|11|37|0.0017%|



There are 123,247 unmatched S1 entities, **5.5848%** of train. The largest observed labelled set has 11 probes. That observation did not become a hard production limit: the prefix selector can select any available prefix length. 

Historical record · 28 September 2026 

7 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 



<!-- Start of picture text -->
Country shift in the supplied data Many probes may match one reference<br>25<br>train $1 5.5848%96 h have no matcheshh<br>20<br>train $2 ><br>=<br>train $3 vis<br>g<br>test S1 5 10<br>test S2 ”<br>5<br>test $3<br>0<br>0 1 2 3 4 5 o 12 3 4 5 6 7 8 § wu<br>Records (millions) True $2/S3 matches per S1<br>Address absence occurs on the probe side<br>What these measurements changed<br>train $1 4 0.00% + France is absent from labelled train<br>train $2 3.36% US/India validation cannot measure its accuracy.<br>+ Names have no literal blanks, but NA/NULL<br>train $3 3.33% tokens require context-aware interpretation<br>* No fixed one-match cap is appropriate for $1<br>test S1 | 0.00% Each probe still has at most one parent.<br>2.85sve and targeted evaluation, not blanket rejection.<br>test s2 + Missing addresses require explicit features<br>test $3 2.68%<br>00 05 #410 415 20 25 30 35 40<br>Literal blank addresses (%)<br><!-- End of picture text -->

_Figure 2. Country composition of reference entities and the distribution of true matches per training S1. France has no labelled training entities._ 

### **4.2 Raw missing values, placeholders and scripts** 

|**Finding from the final audit**|**Design consequence**|
|---|---|
|No empty or whitespace-only<br>business names|A missing-name branch is not supported by the observed raw files, but<br>malformed future inputs still need handling|
|610,389 empty addresses|Preserve an explicit missing-address signal; name-only decisions need<br>additional caution|
|No whole-field address N/A, NULL,<br>NA, NONE or NIL values|Earlier statements about literal NULL addresses were not confirmed by the<br>all-row raw audit|
|132 whole-name NA values and one<br>NULL|“Not blank” does not mean informative; retain raw values and distinguish<br>ambiguous placeholders|
|Both S1 files have Latin-script names<br>and nonempty addresses|Cross-script matching is predominantly probe-side script variation; same-script<br>hard blocking would lose valid pairs|
|113,896 mixed-script names; 966<br>names with no letters|A binary “no ASCII letters” flag is not a complete script detector|
|Maximum field lengths: names up to<br>123 characters, addresses up to 269|Character length is not transformer token length; max_len truncation still needs<br>its own audit|



The final recount corrects the early prose, not the historical code: the normalizer still removes several placeholder-like tokens. A token stoplist and a raw whole-field missing-value policy are not the same operation. `N/A` , for example, is not necessarily handled by the token `na` after punctuation splitting. This needs explicit tests before reuse on a new dataset. 

Historical record · 28 September 2026 

8 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

### **4.3 Early sampled EDA and error-directed work** 

The original Claude analysis recorded zero duplicate S2/S3 ownership across 7.64M ground-truth references and zero cross-country true pairs in a sample of 207,522. The latter supports country partitioning on this dataset, but a sampled observation is not a universal theorem about future business data. 

On an early true-pair sample, name tokens had no overlap in about 14.5% of pairs, addresses in 4.3%, and both simultaneously in only 0.01%. About 18.7% were cross-script under that analysis's definition. These historical sample figures explain the decision to retain both fields; they are not re-labelled here as full-corpus statistics. [E07, E21] 

A blocking-miss audit attributed about 44% of misses to an address absent on one side and 37% to cross-script names. That changed the priority: separate-field lexical retrieval was tried before investing more in transliteration. The two fields fail differently, so doubling down on just one is a poor general strategy. 

### **4.4 Geography: a tempting hard filter that failed** 

The proposal “Telugu implies Andhra Pradesh” was tested as a true-parent exclusion diagnostic. Across two disjoint labelled slices there were 1,140 Telugu true pairs: 246 Andhra Pradesh, 874 Telangana and 20 with unknown/ambiguous region. An Andhra-only rule would remove **441 calibration and 433 reporting true pairs** . It would make search faster by deleting valid answers. [E18] 

The final system therefore does not hard-filter by inferred state or language. It also does not equate a nonempty address with a known state. A future geography channel should add soft evidence or extra candidates with a broad fallback, and measure exclusion before downstream model quality. 

Historical record · 28 September 2026 

9 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **5. Text normalization and learned vocabulary** 

### **5.1 Shared normalization path** 

`textnorm.py` is the common normalization implementation used by lexical retrieval and features. Its steps are NFKC normalization; NFKD-based accent processing; lowercase conversion; explicit allowed-character splitting; removal of selected noise tokens; ordinal suffix normalization; learned Indic-to-Latin token substitution; and learned abbreviation expansion. The core pipeline's candidate files were retained across later feature rebuilds, so sharing a module does not by itself prove an old index was rebuilt with every later normalizer revision. Reproduction should use the frozen candidate artifacts or rebuild and revalidate explicitly. [E08–E10] 

Latin accent handling was a substantial bug fix. A simple ASCII allow-list split “Président” into fragments. A generic Unicode word-character replacement was also unsafe because Indic vowel signs and virama marks are not all word characters. The implemented accent loop removes combining marks following ASCII Latin bases while retaining Indic combining structure, and the tokenizer explicitly admits Latin and U+0900–U+0DFF ranges. 

This fixes the observed Latin/Indic failures; it is **not universal Unicode support** . Cyrillic, Arabic, CJK and other scripts are outside the explicit keep-set. The frozen model should not be advertised as robust to every language merely because the transformer is multilingual. Moreover, `{sas, sa, snc}` are applied as global token stops in the code, despite being motivated by French corpus analysis. Both points belong in the future-generalization plan. 

Deterministic 64-bit token signatures use CRC32-derived bit positions rather than Python's process-salted `hash()` . Bloom overlaps are compressed, collision-prone overlap signals; they are not exact set intersections. This choice made feature generation reproducible across processes without storing a Python token set for every pair. 

### **5.2 Transliteration mined from labelled pairs** 

The transliteration miner samples matched entities, excludes S1 buckets at or above 900, tokenizes cross-script name pairs, and aligns token positions when lengths agree. It votes on Indic-token to Latin-token mappings and retains a winner with at least three votes and confidence at least 0.5. The saved map has **1,314 entries** . Instead of a phonetic approximation such as “yunik,” alignment can recover the actual target spelling “unique.” [E11] 

The early log records 267,502 cross-script pairs examined and 99.9% occurrence coverage within the relevant mining distribution, with 87.2% test Indic name-token coverage in an early audit. A later probe sample measured 93.4% name coverage. These percentages come from different populations and should not be presented as the same metric improving over time without a paired recount. 

An early dictionary accidentally included validation entities. It was re-mined with evaluation buckets excluded; the reported blocking result changed by less than 0.02 percentage points. This correction matters even when the measured impact is small. Buckets 860–899 were still eligible for the miner, so the later 860–900 “holdout” was not untouched by every supervised preprocessing component. 

### **5.3 Abbreviation mining and guards** 

The abbreviation miner uses matching names and addresses with equal token counts. A short form must be at least two characters, strictly shorter than its proposed expansion, share the first character and be a subsequence of the longer form. A candidate needs at least 20 votes and 0.60 confidence among nonidentity candidate expansions. Examples include `pvt→private` , `ltd→limited` , `rd→road` and state-name abbreviations. 

Historical record · 28 September 2026 

10 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

The initial 162-entry map also expanded short function words into US state names, such as `de→delaware` and `la→louisiana` . Two attempted fixes failed: a simple frequency-ratio guard lost useful abbreviations; identity-aware confidence could not reliably separate `in→indiana` from useful entries such as `ltd→limited` . 

The adopted guard checked country-level attestation using unlabeled provided text. For a country with at least 200 occurrences and at least 10% of the short form's sampled usage, the long form must appear in at least 1% as many sampled records. Sampling was capped at 300,000 records per country. The resulting saved map has **149 entries** . [E11] 

The alarming-looking French expansion was initially overdiagnosed. When both sides receive the same near-bijective rename inside a country, set similarities may remain unchanged. Cross-country IDF contamination and genuine same-country collisions are the more specific risks. This was a useful lesson in checking the downstream arithmetic instead of judging normalized text by appearance. 

### **5.4 Shared vocabulary and country-conditioned IDF** 

The v5 feature vocabulary uses **3,304,728 tokens** across **48,458,346 name/address fields** , built from the provided train and test text without test labels. This is transductive feature preparation: it should be disclosed rather than described as training on labelled train data alone. Each country's document frequencies define the IDF used for pairs in that country: 

```
idf(country, token) = max(log(N_country / (1 + df_country(token))), 0)
```

Name and address fields contribute to the field count. Binary token presence yields a set-based TF-IDF cosine: squared IDF weights are summed over shared tokens, then divided by the two weighted set norms. Jaccard remains an unweighted set statistic. Sparse CSR matrices accelerate the same arithmetic as the reference set implementation. 

Global IDF exaggerated the importance of frequent words in smaller-country slices. Earlier diagnostic tables quoted `rue` at global 2.90 versus France-local 1.20; the actual v5 rebuild log reports France-local **1.54** , reflecting the saved field-count convention and corpus. This document does not substitute the earlier illustrative table for the actual model input resource. The mechanism is supported; a country-specific causal leaderboard gain is not separately identified. [E07, E10, E12] 

The first hand-written French legal stoplist `{sarl, sasu, eurl, sci}` had zero observed frequency in the examined French corpus. Measured frequent forms motivated `{sas, sa, snc}` instead. The lesson is to verify proposed preprocessing against the actual corpus and inspect which fields it affects. 

Historical record · 28 September 2026 

11 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **6. Candidate generation and retrieval ablations** 

### **6.1 Deployed lexical blocker** 

Tantivy maintains a separate S1 index for each raw country label. Name and address are separate whitespace-tokenized text fields; an integer reference ID is stored for retrieval. Each probe produces an OR query over up to 128 normalized tokens per field. Name and address have equal boosts **1:1** . The query returns top-10 reference candidates, with zero-based rank and BM25 score. 

This is best described as **separate-field BM25 retrieval** , not an implementation of every possible BM25F formulation. There is no GPU switch for this code path. Worker processes query the CPU index independently. The candidate contract is `(probe_src, probe_eid, s1_eid, rank, bm25)` and rows for one probe must remain together when computing within-probe features. [E08] 

Raw candidate counts were **103,195,092 train** and **99,690,233 test** . Slightly fewer than exactly ten times the probe count is expected where fewer hits are available. Relative score pruning retains `bm25 / max_bm25_for_same_probe >= 0.8` , leaving **17,078,808** core test candidates. It compares parents for the same probe, not probes for the same S1. 

The approximately 17.27 trillion test Cartesian comparisons are reduced to about 17.08 million retained candidates, roughly one millionth of the full cross-product. Candidate recall is the fraction of all labelled true pairs whose parent survives retrieval; the denominator must include positives never returned by the blocker. 

### **6.2 Retrieval experiments and their scope** 

|**Experiment**|**Recorded outcome**|**Interpretation**|
|---|---|---|
|Word tokens vs character<br>3-grams, 300k-reference<br>benchmark|R@50 99.62% vs 98.75%; 737 vs 57<br>queries/sec|Words won in this particular pilot; not a<br>universal n-gram result|
|Unicode fix|Full-index sampled R@10 97.83%→97.97%|Information-preservation bug fix|
|Separate fields|R@10→98.21%; address-missing subset<br>90.0%→95.3%|Avoid penalizing name-only queries against<br>address-padded text|
|Mined transliteration|R@10→98.79%; R@1→96.44%;<br>R@50→99.44%|Recover cross-script lexical matches|
|Relative-score pruning|47.9→7.6 candidates per validation S1 with<br>F0.5 0.9649 unchanged at that stage|Large compute saving on that evaluation;<br>not proof of zero recall loss|
|Rank-only / fixed S1 caps|Rank<1 F0.5 0.9623; cap-10 0.9630|Reduced useful coverage; rejected|
|Deeper k=20|Historical estimate about +0.0007<br>downstream|Deferred; no candidate-count leaderboard<br>penalty exists|



These retrieval measurements use different samples and pipeline versions. For example, the complete v5 train feature file contains 7,541,520 positives out of 7,638,365 ground-truth pairs, about **98.732% raw candidate recall** . That exact full-file quantity should not be equated with the 98.79% sampled earlier result. [E07, E12] 

Historical record · 28 September 2026 

12 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

### **6.3 Field-weight and BM25 length-penalty sweeps** 

The final fine weight test evaluated nine name boosts—1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8 and 2.0—against identical 8,000 probes. Boosts 1.1 and 1.2 changed no outcomes. The best net gain was one recovered probe, with exact paired McNemar p=1.0; boost 2.0 lost two net probes. The coarse address-boost test monotonically worsened R@10 from 98.61% at 1:1 to 97.90% at 1:3. **Equal weights were retained.** The recovered fine grid has nine values, so it would be inaccurate to claim every historic parameter sweep met the later request for at least ten values. [E13] 

Tantivy's production settings remained **k1=1.2, b=0.75** . The Python binding did not expose a production parameter setter in this implementation. An experimental Python reranker rescored the real top-50 pool. At fixed k1=1.2, b=0.30 achieved R@10 98.80% against Python b=0.75 at 98.65% on 4,000 probes. The later joint grid tested ten b values by five k1 values; several cells tied at 98.80%, including b=0.90/k1=0.60. Thus b=0.30 was a promising fixed-k1 result, not a uniquely established global optimum. 

The Python b=0.75 control did not exactly reproduce Tantivy's 98.62%, because query processing and field-length quantization differed. Comparisons within that harness were useful, but substituting it required a new candidate/feature/model evaluation. The first b harness produced 100% at every setting from an inappropriate overlap-selected pool and was discarded as broken. No b change was deployed. [E13] 

Historical record · 28 September 2026 

13 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **7. Exact LightGBM feature dictionary** 

The winning stage-1 feature order is read from `v5_stage1.calib.pkl` and the model text, not inferred from the current feature generator. There are **38 inputs** , not the 45 claimed in an older document. Train Parquet has 41 columns including three metadata columns; test has 40 because it lacks `label` . The later `s1_nm_dup` feature is absent from this model. [E09, E12] 

|**# / frozen input**<br>1.`nm_ratio`|**Family**<br>name|**Definition**<br>Character-sequence similarity on normalized strings (RapidFuzz ratio).|
|---|---|---|
|2.`nm_tok_sort`|name|Ratio after token sorting; reduces word-order sensitivity.|
|3.`nm_tok_set`|name|Token-set ratio; captures overlap with extra or repeated words.|
|4.`nm_partial`|name|Best partial-string ratio; tolerates an extra substring on one side.|
|5.`nm_jw`|name|Jaro-Winkler similarity, on a 0–1 scale.|
|6.`nm_nospace`|name|Ratio after removing spaces; targets joined words/domain stems.|
|7.`ad_ratio`|address|Character-sequence similarity on normalized strings (RapidFuzz ratio).|
|8.`ad_tok_sort`|address|Ratio after token sorting; reduces word-order sensitivity.|
|9.`ad_tok_set`|address|Token-set ratio; captures overlap with extra or repeated words.|
|10.`ad_partial`|address|Best partial-string ratio; tolerates an extra substring on one side.|
|11.`ad_jw`|address|Jaro-Winkler similarity, on a 0–1 scale.|
|12.`ad_nospace`|address|Ratio after removing spaces; targets joined words/domain stems.|
|13.`nm_jaccard`|name|Exact token-set intersection divided by union; zero for empty union.|
|14.`nm_tfidf_cos`|name|Binary-token cosine weighted by country-specific squared IDF.|
|15.`nm_rare_shared`|name|Maximum country IDF among shared tokens; zero if none.|
|16.`nm_postal`|name|Indicator of a shared token matching exactly 5–6 digits.|
|17.`nm_stnum`|name|Indicator that the first digit-only token on each side agrees; not a<br>parsed address.|
|18.`ad_jaccard`|address|Exact token-set intersection divided by union; zero for empty union.|
|19.`ad_tfidf_cos`|address|Binary-token cosine weighted by country-specific squared IDF.|
|20.`ad_rare_shared`|address|Maximum country IDF among shared tokens; zero if none.|
|21.`ad_postal`|address|Indicator of a shared token matching exactly 5–6 digits.|
|22.`ad_stnum`|address|Indicator that the first digit-only token on each side agrees; not a<br>parsed address.|
|23.`bm25`|retrieval /<br>context|Raw separate-field BM25 retrieval score.|
|24.`rank`|retrieval /<br>context|Zero-based rank of S1 among this probe’s retrieved parents.|



Historical record · 28 September 2026 

14 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

|**# / frozen input**|**Family**|**Definition**|
|---|---|---|
|25.`bm25_rel`|retrieval /<br>context|Score divided by the best score for this probe (denominator floor 1e-6).|
|26.`bm25_gap`|retrieval /<br>context|Best score for this probe minus this candidate’s score.|
|27.`n_cand`|retrieval /<br>context|Number of retrieved candidates for this probe in its complete group.|
|28.`tok_overlap`|retrieval /<br>context|Popcount of intersected CRC32-based 64-bit token signatures over<br>name+address.|
|29.`num_overlap`|retrieval /<br>context|Analogous signature overlap for digit sequences over name+address.|
|30.`probe_src`|retrieval /<br>context|Source number 2 or 3; not the probe’s entity ID.|
|31.`nonlatin_s1`|retrieval /<br>context|No ASCII Latin letter in normalized S1 name; nearly/entirely constant<br>here.|
|32.`nonlatin_pr`|retrieval /<br>context|Same Latin-absence test for probe name; not a complete script<br>classifier.|
|33.`script_mismatch`|retrieval /<br>context|XOR of the two Latin-absence flags.|
|34.`addr_empty_s1`|retrieval /<br>context|Normalized S1 address has length zero.|
|35.`addr_empty_pr`|retrieval /<br>context|Normalized probe address has length zero.|
|36.`s1_n_cand`|retrieval /<br>context|All-candidate count reaching S1 divided by mean over seen S1<br>entities.|
|37.`s1_n_rank0`|retrieval /<br>context|Number of probes ranking S1 first, divided by the corresponding file<br>mean.|
|38.`s1_best_rel`|retrieval /<br>context|Current BM25 score divided by the maximum score reaching this S1.|



String ratios generally lie on a 0–100 scale, while RapidFuzz Jaro-Winkler similarity is on 0–1. Trees can handle these distinct scales; pretending all columns were identically normalized would misdescribe training. The `stnum` implementation uses the **first digit-only token found** , not a fully parsed street-number field. The postal flag recognizes a shared 5–6 digit token; it is a heuristic, not a global postal-address parser. 

`s1_n_cand` and `s1_n_rank0` are normalized by means over reference entities receiving candidates in the complete candidate file. `s1_best_rel` compares the current BM25 score with the maximum score received by that S1. Therefore chunk-local recomputation is wrong. Likewise, `bm25_rel` , `bm25_gap` and `n_cand` require the complete candidate list for their probe. Schema equality cannot guarantee semantic feature equality. 

Historical record · 28 September 2026 

15 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

### **7.1 What the saved model used most** 

|**Stage**|**Feature**|**Share of total training gain**|
|---|---|---|
|v5_stage1|rank|55.877%|
|v5_stage1|bm25_rel|16.926%|
|v5_stage1|bm25_gap|12.492%|
|v5_stage1|ad_stnum|2.033%|
|v5_stage1|nm_tok_sort|1.612%|
|v5_stage1|nm_tfidf_cos|1.552%|
|v5_stage1|nm_partial|1.441%|
|v5_stage1|ad_tok_set|1.248%|
|v5_stack|p1|79.847%|
|v5_stack|vs_mean|9.351%|
|v5_stack|p1_rel_max|5.534%|
|v5_stack|p1_share|0.428%|
|v5_stack|s1_n_rank0|0.421%|
|v5_stack|ent_max|0.389%|
|v5_stack|prefix_share|0.368%|
|v5_stack|gap_above|0.217%|



These are training gain shares from the saved models, not causal effects or manually assigned field weights. The dominance of rank and relative BM25 statistics in stage 1 explains the concern about fabricated BM25 scores in dense-only handoff candidates. Stage 2's dominant `p1` , `vs_mean` and `p1_rel_max` show the value of candidate-list context. Correlated inputs can divide gain arbitrarily, so gain alone does not justify dropping a feature without an ablation. 

Historical record · 28 September 2026 

16 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **8. Training splits, LightGBM fitting and calibration** 



<!-- Start of picture text -->
Split responsibilities — the report set was reused<br>Stage 1 fit [0, 400)<br>Historical holdout [860, 900) a<br>Early stopping {900, 250) ||<br>Tree calibration 1950, 975) |<br>Final rule calibration (975, 987) ]<br>Reporting validation [987, 1000) |<br>oO 200 400 600 800 1000<br>Deterministic $1 bucket: (numeric entity ID x 2654435761) mod 1000<br><!-- End of picture text -->

_Figure 3. Label-use boundaries. The final reporting region is disjoint from CE calibration, but repeated inspection and earlier label-derived preprocessing limit the meaning of “held out.”_ 

Every numerical S1 ID receives the stable bucket `(eid * 2654435761) % 1000` . Splitting by S1 keeps that entity's candidate list together. It does not automatically isolate all repeated probe identities, all near-duplicate businesses, or all supervised vocabulary mining. [E06, E11, E12] 

|**Component**|**S1 bucket range / treatment**|
|---|---|
|Stage-1 train|[0,400); retain all positives and 40% of negatives|
|Winning stage-2 train|[400,860); complete candidates after relative-BM25 pruning|
|Later labelled check|[860,900); excluded from v5 trees, but not all vocabulary mining|
|Both tree early-stopping evaluations|[900,950)|
|Tree isotonic calibration|[950,975), unsampled candidates|
|Historical validation|[975, 1000), 54,956 S1 entities|
|CE isotonic fit / final operating-point<br>fitting|[975,987), 26,501 S1 entities|
|Final reporting|[987, 1000), 28,455 S1 entities|



The v5 stage-1 run loaded 18,290,440 training pairs, 16.50% positive after negative subsampling. Its early-stop, calibration and historical validation pair counts were 5,193,568; 2,596,503; and 2,631,482. These are unpruned stage-1 populations. Stage 2 used 7,555,472 training pairs after pruning; corresponding stop/calibration/validation counts were 823,308; 409,855; and 417,336. [E12] 

Historical record · 28 September 2026 

17 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

### **8.1 Saved model hyperparameters** 

|**Parameter**|**Stage 1**|**Stage 2**|
|---|---|---|
|Saved feature count|38|57|
|Saved best iteration|3000|3995|
|objective|binary|binary|
|metric|average_precision|average_precision|
|learning_rate|0.02|0.02|
|num_leaves|127|127|
|min_data_in_leaf|200|200|
|feature_fraction|0.8|0.8|
|bagging_fraction|0.8|0.8|
|bagging_freq|1|1|
|scale_pos_weight|1|1|
|seed|0|0|
|bagging_seed|3|3|
|feature_fraction_seed|2|2|
|num_threads|0|0|
|device_type|cpu|cpu|



Stage 1 requested at most 3,000 rounds with patience 100, and reached the cap at 3,000. Stage 2 requested 4,000 with patience 150; the saved best iteration is 3,995. Both optimize binary log loss and monitor average precision, not macro F0.5. They are CPU LightGBM models. The frozen seeds in the serialized model take precedence over a later script default: notably saved `bagging_seed=3` , while current `model.py` can assign `seed+1` . 

For each stage, isotonic regression fits a monotone map from raw model scores to binary labels on the separate unsampled calibration bucket range. This helps correct score calibration and the training sample's altered class balance. It does not prove conditional calibration by country, address missingness or retrieval channel. 

### **8.2 Why stage 2 is listwise despite a binary loss** 

Stage 1 sees each pair's features. Stage 2 receives those same 38 features plus 19 features computed from the complete S1 list of stage-1 calibrated probabilities. It remains a binary classifier; “listwise” describes the context features, not a ranking-loss objective. 

|**Additional feature**|**Definition**|
|---|---|
|p1|Calibrated stage-1 probability q for this pair.|
|p1_rank|Zero-based rank after sorting q within S1.|
|ent_size|Number of retained candidates for S1.|



Historical record · 28 September 2026 

18 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

|**Additional feature**|**Definition**|
|---|---|
|ent_sum|Sum of q over that entity.|
|ent_max|Maximum q.|
|ent_over05|Count with q>=0.5.|
|p1_rel_max|q divided by entity maximum, with denominator floor.|
|p1_share|q divided by entity probability sum.|
|gap_below|q minus next lower q; last candidate uses q minus zero.|
|gap_above|Previous higher q minus q; zero for leader.|
|prefix_share|Cumulative probability through this row divided by entity sum.|
|mass_after|Entity probability sum minus cumulative probability through this row.|
|big_gap|Largest downward gap within the entity list.|
|cut_rank|One-based position of the largest gap; implementation takes latest tied<br>position.|
|above_cut|Whether this row lies at or above that cut.|
|dist_to_cut|Cut rank minus this row’s one-based rank.|
|ent_second|Second-highest q, zero for a one-candidate list.|
|leader_margin|Maximum minus second-highest q.|
|vs_mean|q minus mean q for the entity.|



For training, stage-2 S1 entities were not used to fit stage-1 trees. This reduces in-sample stack leakage. However, both stages used the same stopping/calibration ranges, and label-derived vocabulary mining used a broader range. An end-to-end independent nested evaluation would require redoing all of these steps within folds. 

Some listwise features are relative, but `ent_size` , `ent_sum` , counts and rank are not invariant to candidate-set size. An old comment claiming that every listwise feature transfers automatically between candidate distributions is too strong. Altering retrieval or pruning changes these inputs and must be revalidated. 

### **8.3 Memory and numerical implementation** 

Arrow row groups are streamed. Negatives and prune filters are applied before materializing full training splits. Float32 matrices are constructed column by column and source arrays released as they are copied. Earlier `column_stack(...).astype(float32)` patterns simultaneously held float64 and float32 matrices and contributed to repeated OOMs. 

Listwise features require complete entity groups even though candidate files are probe-ordered. Production inference used a two-pass layout with stable S1 ordering. Global cumulative sums can have chunk-context-dependent floating-point rounding. In one parity audit, recomputing a subset moved one of 4,496 p2 values by about 0.0011; full-context computation restored agreement below 3e-8. The original score cache and its row order are therefore part of the reproducibility boundary. [E14, E19] 

Historical record · 28 September 2026 

19 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **9. Cross-encoder training and full-coverage scoring** 

### **9.1 Why add a transformer after trees** 

The trees compress each text pair into similarity statistics. That representation cannot retain every word interaction or resolve every cross-script correspondence. A cross-encoder tokenizes both records together and applies attention across them before emitting a match logit. Ditto supplied the record-pair serialization inspiration; this project implements its own fine-tuning/scoring script and does not claim to reproduce Ditto's full augmentation or domain-knowledge pipeline. [R2, E15] 

The selected backbone was `microsoft/mdeberta-v3-base` , approximately 278M parameters including embeddings. XLM-R base was subsequently trained as a diverse second model. The published model repositories identify MIT licensing; their pretrained weights are external model assets, while no external business identity database was used to enrich competition records. The old statement “no pretrained weights are used” is incorrect for the final system. [R4, R5] 

### **9.2 Input representation and training sample** 

Each record is serialized as: 

```
[COL] name [VAL] <raw business_name> [COL] addr [VAL] <raw business_address>
```

The tokenizer receives a left-record/right-record pair. `country` is omitted; lexical normalization and mined maps are not substituted into the transformer text. Maximum combined token length is **128** , with truncation and dynamic padding. This saves memory and compute but creates a possible blind spot for long, subword-heavy records. 

The frozen `ce_pairs_train.parquet` contains **1,582,660 pairs: 791,330 positive and 791,330 negative** . The pair-builder design retains negatives scored at or above a stack p2 hard-negative boundary (default 0.05), samples easier negatives (default 15%), and subsamples positives to match the retained negatives. These are **stack-mined negatives** , not CE-error-mined negatives. The exact saved pair file is authoritative: the original construction was revised from an earlier 3.81M-pair, approximately 79%-positive version before the balanced file was handed off. [E15, E21, E22] 

Balancing avoided spending most updates on easy negative pairs or letting a highly positive pruned sample dominate the training prior. Because this distribution differs from the actual candidate population, its thresholded accuracy is not a deployment metric and its raw sigmoid outputs require calibration. 

Historical record · 28 September 2026 

20 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

### **9.3 Training configuration and split caveat** 

|**Setting**|**Recorded implementation / run**|
|---|---|
|Objective|One output logit; binary cross-entropy with logits|
|Optimizer|AdamW, learning rate 2e-5, weight decay 0.01|
|Schedule|Linear decay; 6% warmup|
|Initial training|Two epochs from pretrained base|
|DGX microbatch|64, selected by the available-memory tier in the script|
|Gradient accumulation|Default 2; effective batch 128 for complete groups|
|Precision|CUDA FP16 autocast and gradient scaler|
|Data loader|Shuffle training slice; two workers; drop final partial batch|
|Internal holdout|First 2% of pair-file rows: 31,653 pairs|
|Scoring|Sigmoid of one logit; commonly batch 128, max_len 128|



The original DGX launch recovered from the imported transcript supplied the local base-model path and otherwise used these script defaults. The first-2%-of-file holdout is a **pair-level** holdout, not a separately enforced S1-group split. Records or entities can recur across its boundary. It is useful for liveness and coarse model monitoring, not as an independent entity-resolution test. The separate S1 reporting buckets used downstream remain the relevant validation boundary. 

At batch 64, there were 24,234 training microbatches per epoch. With accumulation 2, that is 24,234 optimizer updates over both epochs, not 48,468 optimizer updates. Historical references to “steps” often referred to the logged microbatch counter. Full retraining is not guaranteed bit-identical: the script does not comprehensively fix all Torch random seeds and deterministic backend behavior. 

### **9.4 mDeBERTa progression and the rejected v2 expansion** 

|**Checkpoint**|**Internal held-out accuracy**|**Decision**|
|---|---|---|
|Epoch 1|0.9740|Continue|
|Epoch 2|0.9757|Frozen checkpoint used by the winning system|
|Epoch 3|0.9762|Diagnostic continuation only|
|Epoch 4|0.9761|Tiny improvement over epoch 2; not selected for<br>final full scoring|



The extra epochs used the epoch-2 checkpoint at learning rate 1e-5. Their val gate initially improved about 0.9845→0.9847 under the then-current configuration. This was a small effect on a reused split, not a compelling reason to replace the fully scored checkpoint under the deadline. 

A 4,800,020-pair v2 training set was prepared. A pre-authorized diagnostic rule would launch it only for a sufficiently large trained-on versus held-out gap. Measured gaps were 0.0023 for epoch 2 and 0.0034 for epoch 4, below the 0.010 stop boundary. The extra job was therefore canceled. A single low batch loss of 0.0148 had misleadingly suggested near-perfect training; measured mean trained-on BCE was 0.0474 for epoch 2. 

This diagnostic supported canceling that specific job. It does **not** prove that more diverse data, harder negatives or better representations could never help. A small training gap can coexist with underfitting, label noise, domain shift or a deficient feature representation. [E21, E22] 

Historical record · 28 September 2026 

21 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

### **9.5 Coverage mismatch, then full coverage** 

The initial test CE band was p2 in [0.02, 0.98], covering 2,677,471 of the 17,078,808 core test candidates. An early validation script instead used [0.15, 0.85]. It was evaluating a different inference policy. The reported 0.9826 band result was therefore not the exact deployment-matched validation score for the 0.978 submission. 

Re-measuring the deployed band yielded approximately 0.9838 for CE weight 0.8, with an optimum near **weight 0.7, score 0.9841** . The singleton gate then improved the result, and scoring every retained candidate produced the final core validation of 0.985649827274. All **17,078,808** core pairs had mDeBERTa scores in the final system, including pairs the tree stack regarded as highly confident. [E07, E16] 

The full-coverage argument was initially underestimated from US/India validation. Unlabelled French high-p2 rows exhibited many more CE retractions than the seen-country slices, so the narrow-band assumption was especially questionable there. That was evidence of disagreement under distribution shift, not a measurement of French accuracy. 

### **9.6 XLM-R: a failed average that later became useful** 

XLM-R base trained for two epochs in approximately 63 minutes, with internal accuracy 0.9728→0.9743. Full test scoring took about 135 minutes for the same 17,078,808 keys. Both encoders' outputs were checked for row count, key order, duplicates, finite values and probability range. 

The DGX analysis sampled 200,000 French rows from 783,640 with stack p2>0.98. At the mDeBERTa retraction rate of 8.57%, rate-matched XLM-R comparison yielded Jaccard **0.6427** , compared with 0.6430 at a fixed 0.5 cut and 0.6421 at 0.6367. Spearman correlation between encoders was 0.9139 on labelled validation and 0.8288 on this French slice. Those measurements justified testing complementarity but did not demonstrate improved correctness. [E22] 

The globally blended two-encoder submission recorded validation around 0.9844 and leaderboard **0.978** , below the single-encoder full-coverage/gate version at about 0.9856 and 0.980. It was a negative result relative to that incumbent. A lower rank-mean dispersion is not calibrated probability variance, and decorrelation is not a guarantee of useful error cancellation. 

Each encoder needs its own calibrator. A rank mean is not mathematically equivalent to an average of separately calibrated probabilities: different monotone calibration functions weight score regions differently. The final 0.9806 system avoids that failed global averaging policy by using XLM-R only where the conservative admission rule passes. 

Historical record · 28 September 2026 

22 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **10. Core probability blend, prefix selection and ownership** 

### **10.1 Frozen blend** 

Let p2 be the calibrated v5 stack probability, c_M the raw mDeBERTa score and I_M its frozen isotonic map. The winning core uses: 

```
p_core = 0.30 * p2 + 0.70 * I_M(c_M)
gamma  = 1.0                  # no probability-power transformation
beta   = 0.5
gate   = 0.60
```

`ce_calib_v5_full.pkl` records best_w=0.7 and a full [0, 1] band. Its stored f_blend≈0.9845409 predates the additional gate; that metadata number is not the final 0.9858902 score. Calibrator artifacts, coverage and selection settings must travel together. [E12, E16] 

### **10.2 The actual set-selection approximation** 

For one S1, sort candidate probabilities descending. For every prefix length k, compute: 

```
Q(k) = 1.25 * sum(p[1:k]) / (0.25 * sum(p[all]) + k)
Q(0) = product(1 - p[i])
```

Choose the best nonempty prefix only if Q(k)>Q(0) and the highest p reaches 0.60. Otherwise predict empty. For numerically tied prefix scores, prefer the shorter prefix using the implemented 1e-12 tolerance. 

Q(k) is a **ratio-of-expectations approximation** , not the exact expectation of the nonlinear F0.5 ratio. Q(0) equals the empty-set probability only if candidate labels are independent and the modeled universe covers the relevant events; neither is guaranteed here. Missing true candidates and correlated records further limit the interpretation. Work on Bayes-optimal F-measure decisions motivated prefix search, but does not prove this implementation globally optimal. [R3, E06] 

The gate answers whether an S1 should predict any matches. It is different from independently thresholding every pair at 0.60: after an entity passes the gate, its selected prefix may include lower-probability rows. This distinction explains why the gate could help after several ordinary threshold sweeps had failed. 

Historical record · 28 September 2026 

23 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 



<!-- Start of picture text -->
For each S1: complete candidate list<br>Sort calibrated p descending<br>Evaluate every prefix k<br>Q(k)=1.25*prefix_mass/(.25*total_mass+k)<br>Compare with Q(empty)=product(1-p)<br>Best p >= .60 AND<br>best prefix Q > Q(empty)?<br>Keep shortest best prefix Predict empty set<br>All entities: global ownership per probe<br>Highest eligible score wins<br>S1 may keep multiple probes<br>Freeze incumbent assignments<br>Then apply independently validated additions<br><!-- End of picture text -->

_Figure 4. Core decision and ownership flow. The final additions occur after ownership resolution and are not fed through prefix selection again._ 

### **10.3 Global conflict resolution** 

The implemented function groups eligible claims by probe and retains the highest-scoring one globally. An earlier S1 cannot permanently lock out a later higher-score claim. There is no S1 capacity cap, so no Hungarian assignment is required to enforce the observed one-parent-per-probe condition. 

Equal-score ownership ties follow input order in the incumbent function. The production stable S1 ordering and 200,000-entity selection chunks were needed to reproduce the cache-derived output exactly. After correcting an audit's different ordering/chunk boundaries, all **5,670,481** incumbent assignments were reproduced. [E19] 

The highest model score is not necessarily the true owner. Conflict resolution can remove a true claim when a false claim scores higher. Its role is structural validity; any metric benefit has to be measured. Full test selection had **12,602 revoked claims across 8,155 S1 entities** , even though the restricted calibration/report slices had no applicable revoked claims in the backfill experiment. 

Historical record · 28 September 2026 

24 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **11. The final recall improvements** 

The final deadline objective was to raise recall while maintaining the incumbent's already high measured precision. Rather than retrain the full stack, the accepted rules reuse frozen score caches and add narrowly qualified matches. Parameters were chosen on calibration, then checked on reporting, globally and by labelled country. These splits had already influenced many earlier choices, so the result is an internal comparison rather than a new unbiased generalization estimate. [E02, E17] 



<!-- Start of picture text -->
Freeze core output<br>5,670,481 pairs + global occupied-probe set<br>Existing core candidates Previously pruned BM25 candidates<br>p_core >= .30 .6 <= relative score < .8<br>calibrated XLM-R >= .999 pl >= .91; calibrated mDeBERTa >= .995<br>Each branch: unclaimed probes only<br>Highest CE score; tie = lowest S1 ID<br>Union additions<br>Deduplicate identical pairs<br>Veto conflicting new owners<br>Append to incumbent; validate all IDs<br>22,786 + 1,105 = 23,891 additions<br>Report: +120 TP, +0 FP<br><!-- End of picture text -->

_Figure 5. Final admission workflow. Both branches preserve incumbent ownership and cross-family new-owner conflicts are vetoed._ 

### **11.1 Existing candidates: conservative XLM-R admission** 

Eligibility requires all of the following: the probe is globally unclaimed by the core output; the candidate already belongs to the retained core pool; p_core>=0.30; and separately calibrated XLM-R probability >=0.999. Among candidates for the same unclaimed probe, choose the highest calibrated XLM-R score, breaking ties by lower numeric S1 ID. 

A grid of **221 calibration configurations** —17 baseline cuts by 13 XLM-R cuts—selected this rule. Reporting gained **108 TP and zero FP** , with macro F0.5 approximately **0.985860091175** . Full test application added **22,786 pairs** . Its cache-selection pass took about 12.3 seconds and roughly 2.5 GB RAM; the expensive transformer scoring had already completed. 

An isotonic value of 0.999 is an empirical fitted probability, not a demonstrated 99.9% accuracy guarantee in every unseen domain. The high cutoff is useful evidence under the measured calibration regime, but uncertainty remains for the unlabelled test additions. 

Historical record · 28 September 2026 

25 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

### **11.2 Previously pruned lexical candidates: sparse rescue** 

The other branch considers existing raw BM25 candidates with **0.6<=bm25_rel<0.8** . It reuses canonical v5 features and the frozen calibrated stage-1 classifier; no synthetic dense BM25 scores enter this branch. A candidate is admitted only if **p1>=0.91** , calibrated epoch-2 mDeBERTa>= **0.995** , and the probe remains globally unclaimed. Winner/tie rules are analogous to the XLM-R branch. 

The original stage-1 pilot scored 329,136 validation rows; a p1>=0.90 screen retained 29 candidates, 26 true. A **169-configuration** sweep (13 stage-1 cuts by 13 CE cuts) selected the final tighter settings. Reporting gained **12 TP and zero FP** ; calibration gained 14 TP without additional FP. 

On test, Windows screened **8,370,839** rows in about **220.2 seconds** at approximately 1.085 GB peak RSS, retaining 1,245 for CE scoring. DGX scored those 1,245 pairs in **3.93 seconds** , using about **2.45 GB peak GPU allocation** . The accepted rule added **1,105** pairs. This was an example of a small, well-targeted new GPU job producing useful recall without another full rescore. 

### **11.3 Combining the branches** 

The writer preserves the original incumbent sets, deduplicates identical new (S1, probe) suggestions, and vetoes probes with different new owners across the two branches. It asserts that no proposed new probe is already owned and that all receiving S1 IDs exist. No cross-family conflicts occurred in the test output. The combined reporting result was **+120 TP, +0 FP** ; calibration was **+130 TP, +0 FP** . 

The final admissions are not a second unrestricted run of the prefix optimizer. They can increase a set after the core selector has finished, based on their own empirically tested conditions. There is no automatic country-specific override or prior-matching step. 

Historical record · 28 September 2026 

26 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **12. Final validation and leaderboard interpretation** 

|**Reporting metric**|**Full-coverage incumbent**|**Final 0.9806 artifact**|
|---|---|---|
|Pair precision|0.997960960897|**0.997963556140**|
|Pair recall|0.955348610236|**0.956568593563**|
|Entity-macro F0.5|0.985649827274|**0.985890249416**|
|True positives|93,970|**94,090**|
|False positives|192|**192**|
|False negatives|4,392|**4,272**|
|Predicted pairs|94,162|**94,282**|
|Truth pairs / S1 entities|98,362 / 28,455|Same denominators|



The measured recall gain is **0.1220 percentage points** and macro-F0.5 gain **0.02404 percentage points** . Precision increases slightly because the same 192 false positives are divided by a larger number of correct predictions. This establishes the intended trade-off on the measured sample, not the correctness of all 23,891 test additions. [E01–E03] 

|**Country**|**S1**<br>**entities**|**Final precision**|**Final recall**|**Final macro F0.5**|**Added TP / FP**<br>**change**|
|---|---|---|---|---|---|
|India|11,387|0.997705151539|0.962222391819|0.987268198502|+51 / 0|
|US|17,068|0.998137340122|0.952805107277|0.984970943918|+69 / 0|
|France /<br>unseen<br>countries|No labels|Unknown|Unknown|Unknown|Unknown|



An independent dictionary/set-based implementation reproduced the vectorized acceptance metrics. The official full-file validator checked every S1, valid IDs from the 9,969,589 test probes and candidate membership. It completed in about 39.3 seconds with 5.18 GB RSS after its own tests and pilot. Output hashes agreed on both laptops. 

### **12.1 What the 0.00529 validation gap means** 

Validation 0.985890249416 minus reported leaderboard 0.9806 is **0.005290249416** , or **0.5290 percentage points** . The split excludes France, repeatedly reused labels can cause selection optimism, competition across S1 boundaries is restricted, and dataset difficulty/cardinality can shift. All are plausible contributors; their individual causal contributions were not identified. 

Earlier logs inferred French performance around 0.886 by solving `0.961 = 0.850.9742 + 0.15F_France` . That calculation assumes seen-country test accuracy equals validation, the weights are correct for the scored macro population, and there are no other shifts. It is a scenario calculation, not observed French accuracy. Neither a low no-candidate rate nor a high disagreement rate proves good retrieval or matching accuracy: a probe can have many wrong candidates. 

The leaderboard improvement from 0.961 to 0.978 changed preprocessing, IDF and CE behavior together. It cannot be credited independently as +0.017 for IDF and another +0.017 for CE. This document keeps the bundled result and the controlled validation comparisons separate. 

Historical record · 28 September 2026 

27 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **13. Development chronology and accepted milestones** 

This table reconciles the development log with the final artifact. Early rows are historical measurements with their then-current split, feature set and coverage; they are not one perfectly controlled cumulative ablation. “Historical” means recovered from the log/imported discussion, whereas the final row is supported by preserved machine-readable acceptance evidence. [E07, E12, E21] 

|**Stage / experiment**|**Recorded validation**|**Leaderbo**<br>**ard**|**What changed**|
|---|---|---|---|
|Initial edit/string baseline|0.9366|0.924|Establish train-to-output workflow<br>and format|
|Abbreviations + richer<br>token/address features|0.9608|—|Make lexical fields informative|
|Lower learning rate, 127 leaves|0.9649|—|More useful boosting rounds|
|Early 255-leaf stage 1|0.9679|—|Capacity increase; expensive<br>inference|
|First listwise stage 2|0.9723|—|Learn context within each S1<br>candidate list|
|stack3 variant|0.9742|0.961|Stronger early stack|
|v3 / v4 feature variants|0.9739 / 0.9736|—|Shared vocabulary, ordinals,<br>nospace; not capacity-matched to<br>stack3|
|v5 stack|0.9743|—|Country IDF and measured text<br>fixes|
|v5 + initial CE band|0.9826 reported; 0.9838<br>deployment-matched|0.978|Earlier evaluation band was<br>narrower than production|
|Refit CE weight 0.7|0.9841, band policy|—|Correct blend for deployed<br>coverage|
|Add singleton gate|About 0.9850, band policy|—|Protect unmatched entities|
|Full CE coverage + gate|0.985649827274|About<br>0.980<br>reported|Core incumbent for final<br>improvement|
|Global two-encoder average|About 0.9844|0.978|Did not beat single-CE<br>full-coverage policy|
|Conservative XLM-R + sparse<br>rescue|**0.985890249416**|**0.9806**|Final accepted recall improvement|



Historical record · 28 September 2026 

28 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 



<!-- Start of picture text -->
Historical leaderboard milestones<br>99<br>0.9800 = 0.9806<br>98 0.9780 0.9780<br>897<br>x<br>8 0.9610,<br>3 96<br>8<br>2<br>3 95<br>s<br>33<br>5 94<br>Ed<br>93<br>0.92.<br>92<br>Early Fist ce Global Full-coverage Fil<br>Scores arebaselinehistorical/user-reported submissions,stronger stacknot a controlled ablation.band systemthe core score was reported2.CE approximately.system ‘core admissions<br>The 0.9806 final fie is separately identified by its SHA256; no score Is inferred for the later 23:55 artifact.<br><!-- End of picture text -->

_Figure 6. Reported leaderboard milestones. This is a historical sequence, not a causal attribution of each difference to one isolated change._ 

The original user phase plan proposed shared train/test vocabulary, stronger text features, retraining/calibration, France-specific decision tuning and conditional translation. Shared vocabulary, accent handling, nospace similarity, numeric overlap and recalibration were implemented. A full character-ngram feature expansion was avoided for memory/runtime reasons; nospace similarity was a cheap targeted substitute. France-specific beta/prior tuning did not survive validation. NLLB translation was not adopted; the allowed cross-encoder route supplied learned multilingual pair evidence instead. Hybrid dense retrieval remained an evaluated but unshipped train-only branch. 

Historical record · 28 September 2026 

29 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **14. Failed, neutral and incomplete experiments** 

Negative results are bounded by their actual experiment. “This configuration failed” is defensible; “the entire direction is closed forever” usually is not. Several early categorical conclusions were later contradicted by a different, useful implementation of the same broad idea. 

### **14.1 Decision and calibration experiments** 

|**Experiment**|**Measured result**|**Why it was not adopted / lesson**|
|---|---|---|
|Probability scaling / larger<br>selection beta, early model|Recall rose but F0.5 fell; examples<br>0.9649→0.9633 or 0.9635|More returned pairs can cost too much<br>precision|
|Learned per-entity k, 21<br>features and multiclass<br>k=0..12|0.9826→0.9796, about −0.0030|This model/fit objective underperformed;<br>does not prove the oracle gap irreducible|
|Per-country beta|About +0.0001 on seen countries|Weak evidence; no French labels to justify<br>bespoke settings|
|Prior-matched p^gamma<br>recalibration|With gate 0.9856→0.9837; without gate<br>0.9846→0.9832|Forcing a pooled empty-rate target<br>distorted useful probabilities|
|Joint blend/gate, 13×12<br>grid|Cal optimum w=.75/gate=.65 gained<br>slightly on fit and lost on report vs .70/.60|Operating-point overfitting; retain<br>incumbent|
|Deadline gate/gamma/beta<br>sweep, 417 configs|Report F0.5≈0.985545; FP 207 vs 192|Failed precision-preservation objective|
|Deadline calibrated<br>ensemble sweep, 2,288<br>configs|No qualifying improvement over<br>unchanged rule|Complementarity alone was insufficient|
|Singleton gate|Positive; about +0.0009 to +0.0011 by<br>configuration|Accepted despite earlier incorrect dismissal<br>of the whole decision direction|



The training singleton prior of 5.5848% is a descriptive statistic. Similarity between US and India does not prove that France or an arbitrary hidden country has the identical prior. Even with a correct prevalence, the F-optimal prediction rate need not equal it. The final system uses the prior only as a diagnostic, never as a forced target. 

Historical record · 28 September 2026 

30 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

### **14.2 Feature and training experiments** 

|**Experiment**|**Evidence**|**Status**|
|---|---|---|
|More entities with different<br>negative sampling /<br>capacity|Early wide model 0.9631 vs 0.9679|Confounded comparison; no general<br>conclusion that more data hurts|
|Probe-to-probe similarity|Large overall separation; only ~1<br>similarity point between labels in<br>uncertain p2 band|Added complexity not justified by the<br>relevant error population|
|Ten low-gain feature<br>removals|0.0587% combined gain over 2.4M<br>retrieved rows; two features constant|Used in later 28-feature v6; not<br>retroactively removed from winning v5|
|v6 255-leaf /<br>reduced-feature model|Complete checkpoint and cached scores<br>exist; stack ≈0.9748|Global ensemble did not become the<br>0.9806 system|
|scale_pos_weight=.5, v8|Stage 1 AP 0.996601 at 3000;<br>downstream failed for missing calibration<br>file|No completed end-to-end improvement<br>demonstrated; AP is not final precision|
|S1 ambiguity, v9|Feature file completed; recovered training<br>log ends at round 600|Not part of final v5; no validated final<br>precision gain recovered|
|Probe-side ambiguity|S2-only array length conflicted with<br>concatenated S2+S3 indices|Index error; removed|
|Rarest-name-token<br>ambiguity|100% of FP entities vs 98.5% control;<br>~1.02× lift|Nearly universal signal; not useful|
|Same-name S1 ambiguity|75.5% of FP entities vs 40.3% control;<br>1.87× lift|Plausible interaction feature, not a<br>guaranteed standalone fix|



The false-positive autopsy found zero address similarity in 61.5% of incorrect selected pairs versus about 1.3% of correct ones, approximately 46× lift. Yet a blanket no-address rejection would discard both 118 incorrect and 1,244 correct predictions under the recorded analysis. The right lesson is to model the ambiguity interaction and evaluate the trade-off, not delete every record with missing address. [E09, E20] 

### **14.3 GPU hypotheses and their limits** 

Extra mDeBERTa epochs plateaued; the staged v2 expansion did not satisfy its diagnostic launch condition. Country-augmented CE serialization was built and text-parity checked, then abandoned after finding zero France in training and validation and identical left/right countries. A same-country flag would be constant. Adding a country value unseen in fine-tuning could change behavior that US/India validation cannot check. 

That is a sound reason not to deploy the untested change under the deadline. It does not prove that a pretrained encoder cannot understand “France,” nor that the change would necessarily hurt. The missing experiment and missing labels should be stated, rather than replacing them with certainty. 

Larger encoders were considered but not trained. More parameters, ensembling or wider coverage would have consumed the remaining full-test scoring budget. XLM-R base was selected as the affordable diversity test and ultimately repurposed into the successful admissions branch. 

Historical record · 28 September 2026 

31 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

### **14.4 Final rejected recall expansions** 

|**Experiment**|**Reporting effect**|**Decision**|
|---|---|---|
|Wider sparse p1<br>screen>=.1; 529<br>threshold combinations|+22 TP and +1 FP vs original<br>incumbent|Precision declined; rejected|
|Two-CE agreement on<br>wider sparse pool; 324<br>combinations|+11 TP,+2 FP vs accepted final; slight<br>macro-F decrease|Rejected|
|Contention backfill|Mechanism exists on test, but no<br>applicable revocations in restricted<br>labelled slices|No validated precision evidence; not deployed|
|Strict unique<br>name+address exact<br>matches|1,511 labelled matching keys, all<br>already selected|Zero incremental benefit|
|Punctuation-normalized<br>exact variant|2,430 labelled matching keys, all<br>already selected|Zero incremental benefit|



Backfill's proposed mathematical guarantee was wrong: adding an unclaimed false match still reduces precision. The idea remains testable with a global, labelled contention benchmark, but the preserved 0.9806 file does not include it. 

Historical record · 28 September 2026 

32 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **15. Hybrid retrieval handoff: promising, not shipped** 

The teammate supplied `fused_cand_train.parquet` , 1,058,974,382 bytes and **98,930,112 rows in 470 row groups** , with MD5 `032fcb3d24599a4b97da91251d48fed8` . It has the same five-column contract as the local blocker. The reported construction was lexical top-10 plus multilingual-e5-small exact dense top-15, reciprocal-rank fusion and unique exact name/address seeds. There was no complete generating source/version or candidate-origin sidecar in the handoff. [E14] 

The teammate reported train retrieval recall@10 **0.9878→0.9901** with candidates per probe **10.00→9.586** . About 35% of candidates were dense-contributed and carried synthetic `min(bm25)*0.99` values; about 474,662 probes were exact-seeded. Those construction details are attributed to the handoff, not independently reconstructed from missing provenance. 

### **15.1 Why the ready-made feature file was insufficient** 

The supplied v6 feature parts had the same names, order and dtypes as v5 but differed in pair-intrinsic feature values for the same keys. Reporting features were therefore rebuilt using canonical normalization, vocabulary and full-candidate contention. All 30 intrinsic feature checks matched across 174,256 shared pairs after the rebuild. This prevented silently presenting a preprocessing mismatch as a retrieval improvement. 

BM25-related inputs are especially important to the frozen classifier. A fabricated score that preserves a familiar column name is still a different measurement. Future dense integration should carry channel-presence, per-channel ranks, similarities and missingness explicitly, rather than infer origin from a synthetic value. 

### **15.2 Controlled frozen-model comparison** 

A new DGX score run showed up to 0.00684 raw CE drift on 1,280 control pairs relative to the old cached outputs. Both candidate sets were therefore jointly rescored as one 246,167-pair union. The baseline moved by about −0.000005989 macro F0.5 and two true positives, so the paired comparison used the same fresh score pool. 

|**Same reporting entities,**|**Precision**|**Recall**|**Macro F0.5**|**TP / FP**|
|---|---|---|---|---|
|**fresh shared CE pool**|||||
|Incumbent candidates|0.997960918|0.955328277|0.985643838|93,968 /192|
|Canonical fused candidates|0.997806994|0.957524247|0.986311354|94,184 /207|



The raw fused result improved F0.5 but added 15 false positives, failing the user's preference to maintain precision. A subsequent protected-addition policy, selected over **625 configurations** , gained **193 reporting TP with zero additional FP** , reaching **0.986343324966** . It was independently reproduced. That is a promising train-only validation result, not the final leaderboard model. 

There was no fused test file or local builder to create it reproducibly. The root `train_dense_cand.tsv` was a train-only S1-to-probe list, not a scored replacement for the blocker contract. Cached dense test embeddings on the teammate's machine did not provide the missing end-to-end fused artifact. The teammate's runtime estimate was roughly 2.2 hours end to end, beyond the final deadline. No fused result was silently added to the submission. 

Historical record · 28 September 2026 

33 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **16. Engineering workflow and resource placement** 



<!-- Start of picture text -->
Freeze artifacts + commit<br>Check free RAM / VRAM/ physical disk<br>Run separate tests and timed pilot<br>a 7<br>e<br>Old Ubuntu CPU Windows i9 / WSL CPU |<br>Calibration grids 8.37M sparse screen No Co 1060 avaliable<br>Independent metrics / audit 17.08M cached-score selection J 4<br>Allocated DGX H200 MIG<br>Score 1,245 new rescue pairs<br>Frozen checkpoint / tokenizer<br>Lock acceptance rules<br>Precision + recall + macro FO.5<br>Global and labelled-country checks<br>Windows: assemble final TSVs<br>Official format + IDs + membership checks<br>Both laptops: matching output hashes<br>Checkpoint code/evidence to Windows + DGX<br>User submits matching_results.tsv<br><!-- End of picture text -->

_Figure 7. Actual final-stage workflow across machines. The two GPUs are not interchangeable CPU workers, and only useful independent work was scheduled._ 

The old laptop orchestrated small calibration experiments and independent metric checks. The Windows i9 machine did the larger CPU screens, cache selection, output assembly and full ID validation. The allocated DGX GPU scored the small newly required CE batches. The RTX 4060 was available after setup but was not needed for these final few inference jobs. Using every device simultaneously would not itself improve time or score. 

|**Machine**|**Verified resources / role**|
|---|---|
|Old Ubuntu laptop|i5-11400H, 6 cores/12 threads, about 15 GiB system RAM; orchestration and<br>bounded audits|
|Windows laptop / WSL|i9-14900HX, 24 cores/32 threads, 32 GB physical RAM; WSL 24 GB allocation<br>and 8 GB swap; main CPU host|
|RTX 4060 Laptop|About 8 GB VRAM, not 32 GB; CUDA smoke test passed|
|DGX allocated pod|H200 MIG 2g.35gb, about 32.5 GiB visible GPU memory; 16 CPU/32 GiB<br>host-memory quota|



Historical record · 28 September 2026 

34 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

|**Machine**|**Verified resources / role**|
|---|---|
|Shared DGX login node|20 cores/40 threads, 125 GB RAM; not treated as an unrestricted<br>heavy-compute worker|



The actual accelerator was an H200 MIG slice, not the initially assumed A100. A `nvidia-smi` advertised CUDA compatibility number also does not mean an identically versioned CUDA toolkit is installed. These distinctions matter when reproducing kernels or package builds. [E23] 

### **16.1 What was already parallel and what was not** 

RapidFuzz `cpdist(...,workers=-1)` already parallelizes string comparisons in C++. Wrapping a multiprocessing pool around unlimited native threads would oversubscribe cores. The outer feature row-group loop, token preparation and some I/O remain serial. LightGBM training and inference use CPU threads; a CUDA-enabled tree learner does not automatically change `Booster.predict` into GPU inference. 

Benchmarks of 16M RapidFuzz pairs measured about 5.02M/s on the old laptop versus 5.66M/s on the shared DGX host at their respective thread counts, only about 13% faster on DGX. Single-thread RapidFuzz was about 1.89× faster on the laptop. These are workload-specific historical measurements, not an i9 benchmark or a universal CPU ranking. 

Large feature files were about 6 GB; a 5–12 MB/s transfer could cost 10–20 minutes one way. Moving a CPU stage only helps if compute savings exceed serialization, transfer and setup. Compact score caches and disjoint inference shards are usually better candidates for multi-machine work than repeatedly moving the full feature matrix. 

An isolated DGX LightGBM CUDA build passed a five-tree synthetic smoke test, with maximum prediction difference 1.13e-9 from CPU on that tiny test. The Windows packaged build lacked an enabled CUDA tree learner/OpenCL device. Neither result justified silently migrating the production training or promising bit-identical large fits. 

### **16.2 Job survival and memory discipline** 

The later workflow checked free RAM, free GPU memory, disk and active jobs before launch; ran a separate test and representative timed pilot; recorded peak resource usage; committed code or took a recoverable checkpoint; and launched detached jobs with explicit caps. A passing pilot reduces OOM risk but cannot prove a large job safe. 

Several early failures were operational, not model failures: repeated model OOMs, a missing calibrator despite a saved tree file, a stale watcher mistaken for live compute, a queue that deliberately stopped rather than starting the next job, and a full-disk failure after a long score pass. The fixes were explicit state manifests, dependency completion checks, bounded jobs, persistent score caches and correct liveness checks. 

The move to Windows used WSL so the Linux pipeline and path conventions could be preserved. Ethernet and a reconnecting reverse SSH tunnel improved control-channel reliability; detached workers survived control-session loss. Physical Windows free disk was checked separately from the much larger virtual WSL filesystem capacity. Credentials and private connection details are intentionally outside this document. 

Historical record · 28 September 2026 

35 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **17. Validation design, failure modes and uncertainty** 

### **17.1 What was checked well** 

The final acceptance compared identical S1 denominators and included entities with empty truth or no candidates. It measured precision together with recall and macro F0.5, reported country-specific changes, fit rules on calibration before reporting, preserved the incumbent, ran independent metric implementations, verified exact output keys/hashes and used the official format/ID validator. New test rescues used the same frozen CE checkpoint and canonical pair text after parity tests. 

The final writer's synthetic checks covered conflicting new owners, deduplication and incumbent preservation. Its real pilot used about 250 MB RSS; full assembly used 264 MB and 7.3 seconds. Small CPU/GPU pilots made the final short jobs predictable without introducing a training dependency. 

### **17.2 What those checks do not establish** 

1. **Untouched generalization:** reporting buckets were repeatedly inspected across many sweeps. A disjoint calibrator fit is good practice but does not erase prior selection on the reporting data. 

2. **Global ownership realism:** restricting candidate rows by reporting S1 can omit rival parents outside that split. Full-index retrieval and complete probe distractors do not automatically solve this downstream competition issue. 

3. **France or arbitrary-country accuracy:** no labels were available. Match-count, empty-rate and agreement diagnostics cannot be substituted for precision or recall. 

4. **Fold-clean preprocessing:** vocabulary mining excluded buckets>=900, but can touch the 860–900 holdout and stage 2 training entities. A stronger protocol must refit supervised preprocessing within each training fold. 

5. **Exact probability semantics:** isotonic calibration may be coarse in extreme tails and under distribution shift. Candidate truncation and label dependence violate assumptions behind the prefix approximation. 

6. **All additions correct:** zero additional FP on 120 reporting additions is useful finite-sample evidence, not a guarantee for 23,891 test additions. Per-entity clustering and adaptive selection further limit simple uncertainty calculations. 

7. **Causal attribution:** several historical versions changed multiple components at once. Repeatedly assigning the whole leaderboard jump to each component double-counts the same evidence. 

### **17.3 Candidate-export semantics to preserve transparently** 

The shipped candidate file contains the 17,078,808 core retained pairs plus the 1,105 accepted sparse rescues. The rescue branch actually stage 1-screened 8,370,839 rows and CE-scored 1,245. Its export therefore records accepted rescues rather than every intermediate candidate examined by that branch. 

The official validator passed membership checks, but cannot prove that an exported candidate file is an exhaustive record of all pairs scored by every stage. The problem statement describes the last candidate set before final matching. For a strict future code-package reproduction, explicitly classify the stage 1 rescue screen as part of candidate generation and review whether all 1,245 CE inputs—including rejected ones—belong in the companion candidate file. This is an artifact-accounting limitation; it does not change the already submitted matching file or its reported 0.9806. [E03, E04, E17] 

Historical record · 28 September 2026 

36 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **18. Lessons and corrections worth carrying forward** 

|**Earlier statement or practice**|**Correct interpretation for future work**|
|---|---|
|“The decision stage is finished” after<br>several failed sweeps|The singleton gate later helped; a negative family of parameter changes does<br>not exhaust structural changes|
|“All oracle headroom is irreducible”|An oracle is a bound using labels; failure of one learned-k model does not prove<br>impossibility|
|“More data cannot help because<br>train and holdout are close”|A small gap only constrains that diagnostic; diversity, hard negatives and<br>domain coverage remain relevant|
|“Full CE coverage has little value”<br>from US/India validation|Unseen-country disagreement made the extrapolation unreliable; evaluate<br>deployment coverage explicitly|
|“Rank mean equals calibrated mean”|False when the two monotone maps differ|
|“One-to-one only removes false<br>positives”|False: a wrongly high score can revoke the correct owner|
|“Preserving incumbent assignments<br>preserves precision”|False: additions may be incorrect even if nothing is removed|
|“Equal empty rates prove equal<br>domain quality”|Match counts are diagnostics, not labelled quality measures|
|“Candidate count is penalized by the<br>leaderboard”|Incorrect reading; candidate file is not leaderboard-scored|
|“Current code lists all final inputs”|Saved feature metadata defines the frozen model; current code contains later<br>unshipped features|
|“Same Parquet schema means same<br>features”|Values, full-corpus context and preprocessing hashes must also match|
|“A queued monitor means a running<br>job”|Inspect process/service exit state, logs, output completion and dependency<br>gates|
|“Cluster CPU must be faster”|Benchmark the actual parallel kernel and include transfer cost|
|“A smoke test rules out OOM”|It establishes a measured lower-scale case; full-load state still needs projection<br>and limits|
|“Every broader model is<br>automatically more generic”|Overconfident fits and new unsupported input values can regress; validate<br>unseen-domain behavior|



The most reusable principle is to measure the error population that can actually change a decision. Random non-matches made some feature separations look stronger than they were among retrieved hard negatives. Overall probe similarity looked useful but nearly vanished in the uncertain score band. Conversely, small targeted pools of unclaimed probes produced real recall gains when scored with already available models. 

Historical record · 28 September 2026 

37 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **19. Reproducing the winning result** 

There are three distinct reproduction goals: verify the original file, rebuild the same predictions from preserved caches, or retrain the whole system. They require progressively stronger provenance. The frozen outputs and manifests support the first directly. Cache-based rebuilds are practical. A cold retrain needs archived code, resource files, exact model/tokenizer weights and environment control; current defaults alone are insufficient. 

### **19.1 Artifact verification** 

From the project root, this is a read-only identity check: 

```
sha256sum pipeline/output_FINAL_20260927_2328/matching_results.tsv
sha256sum pipeline/output_FINAL_20260927_2328/candidate_pairs.tsv
```

Compare against section 1, then inspect the preserved `official_validation.json` and `validation.json` . A new official validation can be run with the supplied utility's `--matching` , `--candidate` , `--test-dir` and `--check-ids` options. It needs adequate memory; the recorded complete run peaked at 5.18 GB. Do not mistake format validation for recomputing leaderboard quality. 

### **19.2 Required frozen resources** 

|**Role**|**Project-relative artifact**|
|---|---|
|Raw data and scorer|`6ab10eb3b23ba_student_resource/student_resource/`|
|Lexical candidates|`pipeline/work/cand_train.parquet`,`cand_test.parquet`|
|Canonical features|`pipeline/work/feat_train_v5.parquet`,`feat_test_v5.parquet`|
|Text resources|`pipeline/work/translit.json`,`abbrev.json`,`vocab.json`,<br>`vocab_country.pkl`|
|Trees and calibrators|`pipeline/work/v5_stage1.txt`,`v5_stage1.calib.pkl`,`v5_stack.txt`,<br>`v5_stack.calib.pkl`|
|Core CE calibrator|`pipeline/work/ce_calib_v5_full.pkl`|
|XLM-R admission calibrator|`pipeline/work/recall_deadline_20260927_2230/xlmr_topup.json`: frozen<br>`iso2_x`/`iso2_y`interpolation arrays, originally fitted and recorded in<br>`ensemble.json`|
|Labelled score cache|`pipeline/work/val_p2_v5.parquet`; CE val files in`gpu_transfer/`|
|Core test score cache|`pipeline/work/test_scores_full.parquet`|
|mDeBERTa / XLM-R test scores|`pipeline/gpu_transfer/ce_scores_test_full.parquet`,<br>`ce_scores_test_xlmr_full.parquet`|
|Baseline output|`pipeline/output_v5_ceFULL_20260927_0527/`|
|Final additions and evidence|`pipeline/work/recall_deadline_20260927_2230/`; final output directory|
|GPU checkpoints|Preserved DGX`ce_model`,`ce_model_ep34`,`ce_model_xlmr`; only first and third<br>contribute here|



GPU checkpoints must be copied from their preserved location for an offline cold inference package; the code archive and text score files alone are not model weights. The mDeBERTa weight hash 

Historical record · 28 September 2026 

38 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

recorded for final rescue scoring is listed in the hash appendix. Do not silently download the latest base model and call it the same fine-tuned checkpoint. 

### **19.3 Cache-based rebuild sequence** 

1. Restore the matching code commit into a separate directory; verify model/resource/cache hashes. Do not overwrite the archived output. 

2. Recover the core scores in their saved key order. Reproduce stable S1 sorting, 200,000-entity selection chunks, gate 0.60, beta 0.5 and global ownership. Verify the exact 5,670,481 incumbent matches before proceeding. 

3. Apply the locked XLM-R rule with its own frozen calibrator and the complete global occupied-probe set. Do not replace it with the first encoder's calibrator or a rank transform. 

4. Reuse or regenerate canonical sparse-screen probabilities and the same epoch-2 CE scores for the .6–.8 retrieval band. Apply the locked .91/.995 cuts. 

5. Merge same-pair additions once, veto conflicting new owners, preserve baseline rows, and write to a new directory. The recorded final merge consumed 1,105 and 22,786 additions. 

6. Re-run independent metric and format/ID checks. Hash both TSVs and compare them with the historical target. Keep any differences as a new version with an explanation. 

The exact historical build command is preserved in `output_FINAL_20260927_2328/manifest.json` , including Windows path locations, code commit, passed-test and passed-pilot inputs. Use that manifest to resolve inputs; several temporary Windows paths differ from their copied evidence locations on the old laptop. A successful arbitrary invocation of `run_all.sh` is not a guarantee of reconstructing this final multi-branch artifact. 

### **19.4 Training reconstruction and environment** 

`rebuild_v5.sh` records the original canonical feature and tree-training sequence. `crossenc.py` records pair construction, fine-tuning and scoring. `full_cov.sh` records the full-coverage core build. Final additions are implemented by `experiments/conservative_xlmr.py` , `xlmr_test_topup.py` , `sparse_rescue.py` , `sparse_rescue_eval.py` , `final_recall_validate.py` and `build_final_recall.py` . 

Historical environment records show Python 3.13.14, LightGBM 4.7.0, scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1, PyArrow 25.0.1, RapidFuzz 3.14.6 and Tantivy 0.26.2. These are the recorded local versions, not a promise that every wheel is available for every Python/platform combination. Pickled calibrators are version-sensitive; use the archived environment or explicitly verify prediction parity after migration. 

DGX recorded Torch 2.5.0a0+e000cf0ad9.nv24.10, CUDA runtime 12.6 and Transformers 4.46.3. Windows GPU setup used Torch 2.6.0/CUDA 12.6 and the same Transformers version. Different Torch builds, batching and GPU kernels can move scores. Preserve tokenizer files, max_len, autocast precision and calibration. Evaluate the resulting selected pairs if raw numerical parity is imperfect. 

### **19.5 Checkpoints and recoverability** 

The final evidence checkpoint is under `pipeline/work/final_checkpoint_20260927_2337/` , with a complete Git bundle, evidence archive, checkpoint manifest and checksums. At completion, both were copied and checksum-verified on Windows and DGX. The code lives in a nested pipeline Git repository separate from the root migration/documentation repository. 

The complete bundle SHA256 is 

`0dd54db370c353b34fa9c7be40e9e74be2effc013cfd674733fca1b1bcd90992` ; the evidence archive SHA256 is `0417a35d7202e90dbbcf997c9bdd45d8d60c643f6863896e608df76ff2b7de33` . These do not imply that every multi-gigabyte feature file or GPU checkpoint is embedded in the archives: large artifacts remain separately managed, with their identities and locations recorded. 

Historical record · 28 September 2026 

39 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **20. Generalization and the next research phases** 

The winning system already avoids several fragile assumptions: it processes open country strings rather than hard-coding US/India, permits many matches per S1, uses no fixed cardinality prior, and avoids hard language/state exclusions. Nevertheless it is not validated for arbitrary scripts, country distributions, address formats or higher true-match cardinality. A generic future version needs new evidence rather than merely more permissive thresholds. 

### **20.1 Highest-priority validation work** 

Build an entity-grouped, end-to-end evaluation that includes every rival parent for reporting probes. Reserve a genuinely unused final split or use grouped nested cross-validation when the historical splits are exhausted. Re-mine supervised dictionaries within folds. Fit calibrators and tune operating points on separate data, then evaluate a locked configuration once. 

For domain robustness, perform proper leave-country/region-out **refits** , not just slice evaluation of a model trained on those regions. US→India and India→US transfer can reveal fragility, but neither estimates France perfectly. Report script, address-missingness, ambiguity, candidate origin and cardinality strata, with paired per-S1 bootstrap intervals and exact changed-pair counts. 

### **20.2 Best concrete improvement candidates** 

|**Prio**<br>**rity**|**Experiment**|**Why it remains plausible**|**Gate before full deployment**|
|---|---|---|---|
|1|Reproducible fused test<br>retrieval with explicit<br>origin/rank/score<br>features|Protected fused train additions gained<br>193 TP without extra FP|Canonical feature parity; global<br>competitors; seen/unseen-domain refits|
|2|Candidate-channel-awa<br>re reranker|Current trees depend strongly on<br>lexical rank/BM25; fake dense scores<br>are mis-specified|Compare frozen ranker versus retrained<br>ranker on the same candidates|
|3|Missing-address ×<br>S1-ambiguity<br>interaction|Measured error concentration; no<br>validated final ambiguity run yet|Complete training/calibration, compare<br>precision and recall together|
|4|Universal-script<br>normalization and safe<br>fallback|Explicit keep-set can delete unseen<br>scripts|Information-loss tests across all<br>observed/new scripts, then rebuild both<br>sides|
|5|Global<br>contention/backfill with<br>proper labels|12,602 test revocations indicate a real<br>mechanism|Include out-of-split rivals; validate<br>additions and any reassignments|
|6|BM25 b/k1 and<br>multi-channel depth|Small paired retrieval headroom exists|Paired full-index retrieval and<br>final-model benefit, not R@k alone|
|7|Better CE sampling or<br>representation|Small train gap does not close<br>hard-negative/domain-diversity work|Entity-group split; country/length<br>coverage; full calibration and scoring<br>budget|



Keep core matching, candidate generation and decision changes separable in experiments. Evaluate sparse-only, dense-only, union, RRF and exact-seed legs independently, then together. Record exact RRF constants, channel weights, deduplication and tie rules. A one-candidate exact seed can improve efficiency but also hide an alternative true parent if its uniqueness premise fails. 

Historical record · 28 September 2026 

40 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

### **20.3 Making future inputs safer without claiming free score** 

Add script detection that distinguishes no-letter, mixed-script and specific-script inputs. Replace accidental fallback of unknown country to an existing IDF row with an explicit, evaluated fallback. Audit global legal-form stops and ambiguous name tokens such as NA before expanding the stoplist. Add a guarded name-only retrieval channel for missing addresses rather than assigning a geography from language alone. 

Test max_len sensitivity and truncation location on long records. Test higher-cardinality synthetic or held-out entities without imposing the training maximum 11. Calibrated probabilities should be stress-tested by missing-address and channel strata; pooled tail calibration can be particularly brittle at cuts .995 and .999. 

These are proposed future phases, not completed work or a promise of 0.991. The public result remains 0.9806. The purpose is to produce trustworthy additional recall and reduce the validation-to-test mismatch, rather than optimize a repeatedly inspected number. 

Historical record · 28 September 2026 

41 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **21. Provenance, attribution and remaining gaps** 

### **21.1 What was recovered from Claude** 

The imported **“Model selection guide”** conversations contain the original EDA, retrieval-direction decision, Unicode and deterministic-hash fixes, vocabulary mining, early baseline training, listwise stack development, phase-plan discussion and later threshold/coverage experiments. The **“DGX access and GPU testing”** conversations contain CE launch commands, epoch results, full-score assembly and resource diagnostics. 

Several same-title imports repeat earlier turns and diverge at later messages. They are overlapping snapshots, not nine independent experiments and not necessarily nine consecutive context-compaction segments. This reconstruction deduplicated evidence conceptually and preferred saved artifacts/logs when prose conflicted. Claude's earlier contribution was sufficient to document the model's origin without asking the user to re-explain three days of work. 

Attribution: the user and teammate supplied phase ideas and retrieval/geography proposals; the original Claude work built and evolved the lexical/tree/CE pipeline; the DGX Claude work executed transformer training/scoring and diagnostics; the later Codex work audited the handoff, corrected feature/parity and validation assumptions, coordinated deadline jobs and assembled the 0.9806 conservative-recall artifact. The singleton gate was credited in the historical log to the teammate's pipeline. 

### **21.2 Evidence confidence and honest unknowns** 

**Artifact-backed:** final file identity, tree feature lists/iterations, saved operating points, source-file audit, final validation and output counts. **Historical measured:** older log tables and imported tool results, subject to their stated samples/configurations. **User-reported:** leaderboard 0.9806 and some collaborator observations. **Hypothesis or deferred work:** isolated French accuracy, true cause of the entire leaderboard gap, future hybrid-test gains and claims of universal generalization. 

The original HF base-model revision IDs and a complete deterministic training manifest were not consistently stored in local final records. Fine-tuned checkpoint hashes are stronger inference identifiers than an unpinned model name. The fused retrieval builder/version and its candidate-origin sidecar are missing, but fused retrieval is not part of this winning model. Incomplete v8/v9 runs lack a recovered final end-to-end score; no success is invented for them. 

### **21.3 Optional archival prompt for Claude or the teammate** 

No reply is needed to understand the verified 0.9806 methodology. For a stricter cold-retraining archive, the following focused request would fill remaining provenance gaps: 

Please provide artifacts or exact log excerpts, not reconstructed guesses: (1) the original mDeBERTa and XLM-R base-model revision IDs, fine-tuning launch scripts, Torch seed settings and final checkpoint hashes; (2) the exact command and script revision that created the final balanced 1,582,660-pair CE training file, including whether it was shuffled before the first 2% holdout split; (3) the actual result/exit status of the v8 class-weight and v9 ambiguity runs, if they completed elsewhere; (4) the fused train-candidate builder, all RRF/depth/seed settings, channel-origin sidecar and any matching test artifact. Separate measured facts from estimates. Do not run training or alter the submitted 0.9806 files. 

The document is complete for the preserved winning inference method; these requests would strengthen cold-retraining reproducibility and the unshipped research archive rather than change the submitted model description. 

Historical record · 28 September 2026 

42 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **22. Reconciliation of nearby artifacts** 

|**Artifact / result**|**Correct interpretation**|
|---|---|
|`output_v5_ce_20260927_0033`<br>/0.978|Earlier band-coverage CE system, not final full coverage|
|`output_2enc_20260927_1508`/0.978|Earlier global two-encoder combination, not final conservative XLM-R rule|
|`output_v5_ceFULL_20260927_0527`<br>/about 0.980|Full-coverage epoch 2 core incumbent|
|Protected fused train validation<br>0.9863433|Stronger internal train-only result; no deployable matching test candidates|
|`output_FINAL_20260927_2328`<br>/0.9806|The subject of this document|
|`output_FINAL_20260927_2355`|Later v6 admission extension; unknown leaderboard result|



The 23:55 extension added 1,097 test pairs after an 18-cutoff calibration sweep on a cached v6+two-CE score. Its reporting delta over the 0.9806 artifact was only **+2 TP, 0 FP** , reaching macro F0.5 **0.985895117739** , precision 0.997963599338 and recall 0.956588926618. Its matching-file SHA256 is `f9aeb180608d88cae8596d1b089fc97135ec35067b0d116f2c1bbd22a70b0355` . It is preserved as a separate candidate, not silently labelled the 0.9806 submission. [E24] 

Historical record · 28 September 2026 

43 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **23. Artifact hashes and evidence index** 

Paths in this index are relative to the project root `/home/kartikey/Downloads/Amazon ML challenge` . Source files are interpreted with their recorded commit where available; current comments may include later additions. The source ledger is intended to make the document auditable without reproducing multi-gigabyte data in the document itself. 

|**Artifact**|**Recorded / verified SHA256**|
|---|---|
|pipeline/work/v5_stage1.txt|7d99d519beb944529f526d28d8e13782c8757baf6e9f9c62628a6f95ddeb5<br>8da|
|pipeline/work/v5_stage1.calib.pkl|9e7f51d2bf5eb6de96c37ff721e04fa50832356c6bb460ee198fbdcc0b842d<br>4c|
|pipeline/work/v5_stack.txt|bdf371960079f1c87bb7edd8dc567abe50fcfb29bb17413cf23c00238916ab<br>55|
|pipeline/work/v5_stack.calib.pkl|648758e96ce6b8d4a3006e5e969a90ccdf7c211233befd2736fd9292d5f45<br>b5b|
|pipeline/work/translit.json|6994bd59543998a703d0b97332af6ddc5b682b89dad63c93100308d848c6<br>2165|
|pipeline/work/abbrev.json|22a12ef8074ad92fe99cedb35e2e1a0b8697298e6c13f9fce5e6b2311e056<br>97d|
|pipeline/work/ce_calib_v5_full.pkl|b142462126605df0a3c1c68e0f029b259f05c9b2a32a286bcedfac7e47bbb<br>33a|
|pipeline/work/recall_deadline_20260927_2<br>230/xlmr_topup.json|5d58345f3c1021b01d6a5f88b263a8daceb9eb1e3070ecbfe51f617303269<br>fa2|
|DGX ce_model/model.safetensors|88e18f98dd267affa0f7ba6ea107a8567ae9d75a5f28919f08d92bd1cb43e1<br>04|
|Final added_pairs.parquet|1deae54087d58401528bd72c9814c1c76a24777b1e5b62c172e84cdb6d8<br>e7d0f|
|train_source1.tsv|591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d149<br>73|
|train_source2.tsv|6336c1a055eec79cf8a6d99fdc8d32a2e4d9dc2662e00963cb35d66b89ed0<br>9ed|
|train_source3.tsv|67da22f5151898ff3006febd836c1a159e97ae95efa7257a5aff4fda685e58e<br>9|
|test_source1.tsv|3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2<br>f5|
|test_source2.tsv|79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7<br>b2dd|
|test_source3.tsv|850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda<br>035|



|**ID**|**Source**|**Supports**|
|---|---|---|
|E01|migration/final_submission_20260927.json|Final identity, exact metrics, user-reported<br>leaderboard and backup status.|



Historical record · 28 September 2026 

44 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

|**ID**|**Source**|**Supports**|
|---|---|---|
|E02|pipeline/experiments/FINAL_SUBMISSION_20260927.m<br>d; pipeline/work/recall_deadline_20260927_2230/final_<br>validate.json|Accepted rules, independent metrics and final<br>negative results.|
|E03|pipeline/output_FINAL_20260927_2328/{manifest,vali<br>dation,official_validation}.json|Build command, inputs, output hashes, counts and<br>validator evidence.|
|E04|6ab10eb3b23ba_student_resource/student_resource/R<br>EADME.md; utils/validate_submission.py|Organizer task, metric and TSV/candidate-file<br>contracts.|
|E05|pipeline/work/eda_plan_20260927_2220/{full,recount_<br>v2}.json; train_ground_truth.tsv|Two-parser source audit; streaming ground-truth<br>cardinality recount for this document.|
|E06|pipeline/model.py|Bucket split, prefix approximation, ownership, metric<br>and stage1 training.|
|E07|pipeline/experiments/log.md|RUN-001 onward: historical measurements,<br>corrections and negative results.|
|E08|pipeline/blocking.py|Country indexes, retrieval direction, query fields,<br>boosts and candidate contract.|
|E09|pipeline/features.py|Feature definitions, sparse calculations, full-file<br>context and later ambiguity implementation.|
|E10|pipeline/textnorm.py; pipeline/build_vocab.py;<br>pipeline/work/vocab_country.pkl|Normalizer, saved token resources and<br>country-conditioned IDF.|
|E11|pipeline/mine_translit.py; pipeline/mine_abbrev.py;<br>work/{translit,abbrev}.json|Supervised vocabulary mining, exclusion boundary<br>and attestation guard.|
|E12|pipeline/work/{v5_stage1,v5_stack}.txt and .calib.pkl;<br>pipeline/work/rebuild_v5.log; pipeline/rebuild_v5.sh|Authoritative model inputs/parameters/iterations;<br>training counts and historical launch flow.|
|E13|pipeline/work/{boost_fine,boost_sweep,b_test,bk1,join<br>t_wg}.log|Field-weight, BM25 reranking and operating-point<br>grids.|
|E14|pipeline/experiments/FUSED_HANDOFF_RESULT.md;<br>RECALL_DEADLINE_RESULT.md;<br>pipeline/work/fused_audit_20260927_local/|Canonical feature mismatch, fresh CE comparison,<br>fused provenance and result.|
|E15|pipeline/crossenc.py; pipeline/GPU_HANDOFF.md;<br>pipeline/work/ce_pairs_train.parquet|CE serialization, sample, train/score implementation<br>and handoff.|
|E16|pipeline/ce_blend.py; full_cov.sh; submit_ensemble.py;<br>work/ce_calib_v5_full.pkl|Coverage, calibrated blending, production grouping<br>and core output.|
|E17|pipeline/experiments/{conservative_xlmr,xlmr_test_to<br>pup,sparse_rescue,sparse_rescue_eval,build_final_reca<br>ll}.py; work/recall_deadline_20260927_2230/|Admission algorithms, locked grids, score manifests,<br>resources and final merge.|
|E18|pipeline/work/recall_deadline_20260927_2230/geogra<br>phy.json|Telugu/state exclusion counts in calibration and<br>report slices.|
|E19|pipeline/work/recall_deadline_20260927_2230/window<br>s_runtime/contention_v2_full.json;<br>stack_smoke_fullctx.json|Production parity, global revocations and full-context<br>feature verification.|
|E20|pipeline/work/{v7,v8_prec,v9,prec_chain}.log;<br>prec_chain.sh; v9_ambig.sh|Incomplete precision/ambiguity runs and execution<br>failures.|



Historical record · 28 September 2026 

45 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

|**ID**|**Source**|**Supports**|
|---|---|---|
|E21|Imported “Model selection guide”:<br>01a0e328-1c1d-7dd2-812a-b322d0c0330f;<br>01a0e328-1a57-77f3-b70b-9fbeb453fefd;<br>01a0e328-1962-7313-a83a-5b3d4624aacf|Original EDA/methodology, user phase plan,<br>construction history and latest parameter<br>corrections; overlapping snapshots.|
|E22|Imported “DGX access and GPU testing”:<br>01a0e328-1979-71c3-9eaa-ea8e292c47fb;<br>01a0e328-1a13-7372-92c1-a5abfc4ce56d|Training launch commands, epochs, diagnostic gaps,<br>full-score completion and France agreement<br>measurements.|
|E23|pipeline/experiments/GPU_EXECUTION_AUDIT.md;<br>EDA_IMPROVEMENT_PLAN.md; migration/|Verified hardware, CUDA smoke tests, numerical<br>parity, resource and migration workflow.|
|E24|pipeline/experiments/FINAL_EXTENSION_20260927.md<br>; migration/final_extension_20260927.json|Later 23:55 artifact; kept separate from the reported<br>0.9806 result.|



The document's companion editable source and machine-readable evidence extract are preserved beside the rendered document. They are documentation artifacts, not dependencies added to the ML pipeline. Rendering installed no packages into the production ML environments and ran no model training. 

Historical record · 28 September 2026 

46 

AMAZON ML CHALLENGE 2026  |  0.9806 SUBMISSION METHODOLOGY 

## **24. Research references and terminology** 

**R1.** Paulsen, Govind and Doan, _Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching_ , PVLDB 16(6), 2023. Top-k inverted-index blocking and retrieval direction informed this implementation. Publisher paper. 

**R2.** Li et al., _Deep Entity Matching with Pre-Trained Language Models_ (Ditto). Record serialization and joint pair encoding informed the CE design; the local script does not implement all Ditto components. Author paper. 

**R3.** Waegeman et al., _On the Bayes-Optimality of F-Measure Maximizers_ , JMLR 15, 2014. Motivation for set-level F-measure decisions; not a proof that the local ratio-of-expectations selector is exact. Journal article. 

**R4.** Microsoft mDeBERTa-v3-base model repository: multilingual base checkpoint and MIT license declaration. Model repository. 

**R5.** FacebookAI XLM-RoBERTa-base model repository: multilingual base checkpoint and MIT license declaration. Model repository. 

|**Term**|**Meaning in this project**|
|---|---|
|S1 / reference|Deduplicated business entity whose match set is predicted|
|Probe|One S2 or S3 record querying the S1 index|
|Blocking / retrieval|Candidate generation before final match selection|
|p1 / p2|Calibrated stage 1 / stage 2 LightGBM probabilities|
|CE|Transformer cross-encoder that reads the two records jointly|
|Isotonic calibration|Fitted monotone score-to-probability mapping|
|Listwise features|Context derived from an S1's complete candidate score list|
|Singleton gate|Entity-level test of whether any match may be predicted|
|Singleton|Historical shorthand for an S1 with zero true S2/S3 matches|
|Retraction|A model removes/rejects a previously confident candidate; not automatically a<br>corrected error|
|Protected addition|New pair using an unclaimed probe while preserving existing owners|
|RRF|Reciprocal-rank fusion of retrieval lists; unshipped train-only branch here|
|Macro F0.5|Mean of entity-level F0.5, including empty-truth entities|
|Pair-micro precision/recall|Aggregate pair counts, distinct from macro F0.5|
|OOM|Out-of-memory failure; operational result, not a score verdict|



**Record boundary:** submission development through 27 September 2026; documentation verified 28 September 2026. No unobserved leaderboard value, new training result or hidden-country performance is asserted. 

Historical record · 28 September 2026 

47 

