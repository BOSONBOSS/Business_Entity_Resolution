# Project Proposal & Evaluation Report
### CSE2032 — Machine Learning | BML Munjal University | 2026–27

**Project Title:** Business Entity Resolution across Noisy Multi-Source Records
**Team Members:** 
* Ananshi Sharma (240377)
* Aditya Nadar (240368)
* Sumit Singh (240344)
* Dhruv Sharma (240444)

---

## 1. Problem Statement
In modern data systems, real-world businesses are frequently registered across multiple, independent databases (e.g., taxation records, food delivery platforms, local registries) without a shared primary key. 

The problem is to perform **Business Entity Resolution** on a massive scale (24.2 million records). Given a deduplicated reference dataset (Source 1) and two noisy, unlinked probe databases (Source 2 and Source 3), the objective is to build a Machine Learning pipeline that automatically determines which records refer to the identical real-world business. The system must overcome extreme text noise including abbreviations, missing addresses, website domain names, and cross-script transliteration (e.g., Devanagari to Latin). The model's success is measured by the **Macro F₀.₅ score**, which heavily penalizes false merges (false positives).

## 2. Objectives
1. **Scalability:** Design a pipeline capable of reducing 17.2 trillion raw comparisons into a computationally feasible subset using a high-recall blocking strategy.
2. **Precision-First ML:** Train a binary classifier (LightGBM) optimized specifically for high precision, given the F₀.₅ metric's 2x penalty for false positives.
3. **Out-Of-Distribution (OOD) Generalization:** Build features that are language-agnostic to handle unseen test domains (e.g., France data, which makes up 15% of the test set but 0% of the training set).
4. **End-to-End Automation:** Deliver a pipeline that takes raw unlinked TSV files and outputs a formatted `matching_results.tsv` file representing a completely resolved bipartite graph of entities.

---

## 3. Exploratory Data Analysis (EDA) — Detailed Results
Before architecting the pipeline, we conducted an exhaustive EDA on the 24.2 million records. Our core findings dictate our methodology:

1. **The Scale & Singleton Trap:** 5.58% of the training reference entities have zero true matches. Because the metric is macro-averaged, correctly predicting "no match" for singletons is worth 1.0 points. Over-matching is mathematically disastrous.
2. **Ambiguous Names:** 48% of businesses in Source 1 share their exact name with another distinct business (e.g., multiple "Star Services"). *Result:* Name similarity alone is a weak signal; address must act as the primary discriminator.
3. **Missing Addresses:** 168,967 records (~3.3%) in Source 2 have absolutely no address (often written as literal strings like "null" or "-"). *Result:* Missing addresses are over-represented in false matches (1 in 8 distractors).
4. **Website Domain Noise:** ~4% of S2/S3 records list their website (e.g., `dhaninvestments.com`) instead of their name. *Result:* Standard word-token similarity is 0% for these cases.
5. **Non-Latin Scripts:** Many Source 2 records are in Devanagari, Tamil, Telugu, etc. *Result:* French records (`é, à`) are occasionally falsely flagged as non-Latin due to encoding bugs, requiring pre-processing accent removal.
6. **Positional Digits:** ZIP codes are highly discriminative. However, `123 10001` and `123 10002` share 50% of their digits despite being different locations. *Result:* Positional extraction (first number = building, last number = ZIP) is required.
7. **Bipartite Truth:** If an S2 record matches an S1 entity, it *never* matches a second S1 entity. 

---

## 4. Methodology & Phase-Wise Division of Work
Based on our EDA and learnings, we will implement a Two-Stage Pipeline (Blocking + ML Classification). The methodology is broken down into 6 execution phases:

### Phase 1: EDA & Baseline Setup *(Completed)*
* Parsing the 24.2M TSV files, conducting quote-aware parsing.
* Extracting the 7 core EDA rules listed above.

### Phase 2: Text Normalization & Preprocessing
* **Methodology:** Clean text to maximize matching potential before ML.
* Strip French accents (`unicodedata.normalize('NFKD')`) to prevent non-Latin misclassification.
* Remove legal suffixes (Pvt, Ltd, SARL, SAS) to prevent false inflation of Jaccard similarities.
* Convert literal "null", "n/a", and "-" strings into actual empty spaces.

### Phase 3: Candidate Generation (BM25 Blocking)
* **Methodology:** Brute force is O(N×M) (17 trillion pairs). We will build a **Tantivy BM25 inverted index** on Source 1.
* Query S2 and S3 against S1 using separate name and address fields (1:1 boost).
* **Relative Pruning:** Keep only top-10 candidates whose BM25 score is ≥ 80% of the best score for that probe. *This reduces the problem space by 1,000,000× (to 17 million pairs).*

### Phase 4: Feature Engineering
* **Methodology:** Compute 38 statistical features for the 17M candidate pairs.
* **Token Metrics:** Jaccard similarity, TF-IDF cosine (fit on test+train to fix France OOV words), RapidFuzz edit ratios.
* **N-Grams:** Character trigrams (to catch domain name noise like `dha`, `han` for `dhaninvestments.com`).
* **Numeric Overlap:** Positional extraction to compare ZIP codes separately from building numbers.
* **Context Features:** `both_addresses_missing` flag, BM25 rank gap.

### Phase 5: Machine Learning Classification (LightGBM)
* **Methodology:** Train a binary Gradient Boosted Tree (LightGBM) on the engineered pairs.
* LightGBM naturally handles missing values and mixed feature scales (ratios 0-1 vs BM25 0-20).
* **Listwise Context:** Pass the Stage 1 predictions into a Stage 2 model that looks at the *entire* candidate list for a single S1 entity to rank them relative to each other.

### Phase 6: Probability Calibration & Decision Thresholding
* **Methodology:** Use Isotonic Regression to calibrate raw LightGBM scores into true probabilities.
* Implement the **Lewis 1995 Expected F₀.₅ algorithm** to select the optimal prefix (number of matches) per entity, heavily favoring precision.
* Enforce **Global Mutual Exclusivity** (if S2 belongs to two S1s, drop the lower probability claim).

---

## 5. Member-Wise Division of Work

| Team Member | Primary Responsibilities |
|---|---|
| **Ananshi Sharma (240377)** | **Phase 1 & Phase 2:** Data Profiling, Text Normalization pipeline. Responsible for implementing accent stripping, legal suffix removal, and the `is_non_latin` detection flags. |
| **Aditya Nadar (240368)** | **Phase 3:** Candidate Generation (Blocking). Responsible for building the BM25 index, optimizing queries, and ensuring the 1,000,000x reduction without dropping true matches. |
| **Sumit Singh (240344)** | **Phase 4:** Feature Engineering. Responsible for programming the 38 features (Character N-grams, TF-IDF cosines, and positional ZIP code extraction algorithms). |
| **Dhruv Sharma (240444)** | **Phase 5 & Phase 6:** ML Modeling & Thresholding. Responsible for LightGBM model training, Isotonic Calibration, and implementing the Lewis 1995 F₀.₅ maximization algorithm. |

*Note: All members will collaborate on the final script integration, testing, and report writing.*

---

## 6. Literature Review & Existing Gaps

1. **Paulsen et al. (2023) - *Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching***
   * *Contribution:* Proves that simple inverted-index blocking (like our Phase 3) outperforms many deep learning blockings in scale and accuracy.
   * *Gap Identified:* Sparkly struggles with cross-script entity matching (e.g., Hindi vs English) unless an external translation API is used. Our project bridges this by relying heavily on positional numeric features (addresses) when scripts mismatch.
2. **Lewis, D.D. (1995) - *Evaluating and Optimizing Autonomous Text Classification Systems***
   * *Contribution:* Provides the mathematical foundation for maximizing F-measure based on expected probability distributions.
   * *Gap Identified:* Assumes perfectly calibrated probabilities. We address this gap by adding Isotonic Regression before applying Lewis's prefix selector.
3. **Li et al. (2020) - *Deep Entity Matching with Pre-Trained Language Models (Ditto)***
   * *Contribution:* State-of-the-art transformer architecture for entity matching.
   * *Gap Identified:* Computationally impossible to run on 17 trillion pairs. We use their insights conceptually but replace the transformer with a faster LightGBM tree stack on statistical features for scalability.

---

## 7. Final Deliverables

At the conclusion of the semester, the team will submit:
1. **Source Code Repository:** Containing modular, clean scripts for each phase (Normalization, Blocking, Features, Model, Decide).
2. **Trained Models:** Serialized LightGBM models (`.txt`) and Calibration wrappers (`.pkl`).
3. **Inference Script:** A fully automated script (`submission.py`) that ingests raw TSV datasets and exports the final `matching_results.tsv`.
4. **Final Project Report & Presentation:** Documenting our validation scores, ROC curves, feature importance charts, and final leaderboard metric (Macro F₀.₅).
