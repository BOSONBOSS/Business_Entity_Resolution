# Comparative Machine Learning Approaches for Business Entity Resolution

**Course:** CSE2032 — Machine Learning | **Semester:** 5th Semester, B.Tech CSE  
**Academic Year:** 2026–27 | **Institution:** BML Munjal University  
**Team Members:**
- Ananshi Sharma (240377)
- Aditya Nadar (240368)
- Sumit Singh (240344)
- Dhruv Sharma (240444)

---

## 📁 Repository Organization

The repository is structured into clear, purpose-driven directories:

```
.
├── 01_submission_deliverables/        # Official academic deliverables & templates
│   ├── Final_Project_Report_Template.docx  # Official 27-page university report template
│   ├── ML_Project_Master_Plan.md           # Approved 10-phase master implementation plan
│   ├── ML_Project_Master_Plan.pdf          # Formatted PDF of the master plan
│   ├── ML_Project_Proposal_Report.md       # Midterm 25-mark project proposal report
│   ├── ML_Project_Proposal_Report.pdf      # Formatted PDF of proposal report
│   └── ML_Project_Evaluation_Doc.md        # Faculty viva preparation & pitch guide
│
├── 02_academic_references/            # Ground-truth academic & domain reference material
│   ├── ML.pdf / ML_handout.md              # Official CSE2032 course syllabus & evaluation rubric
│   ├── Amazon_ML_Learnings_doc.md (.pdf)   # 24-chapter engineering record of 0.9806 system
│   ├── 01_EDA_business_entity_resolution.html (.pdf) # Rendered EDA reports with charts
│   ├── html-to-markdown.md.txt             # Raw notebook outputs & distributions
│   └── PROBLEM_STATEMENT.md                # Formal entity resolution task specification
│
├── 03_historical_plans_and_notes/     # Historical sprint roadmaps & competitive notes
│   ├── FINAL_PLAN.md                       # Validation gap & OOD analysis
│   ├── MASTER_OPTIMIZATION_PLAN.md         # Loss function & feature tuning notes
│   ├── PREPROCESSING_PLAN.md               # Early 10-step cleaning blueprint
│   ├── WINNER_SOLUTIONS_REPORT.md          # Competitive ML research & cross-pollination
│   ├── CLAUDE.md / HACKATHON_RULES.md      # Historical competition constraints
│   ├── SUBMISSION_DOC_GUIDE.md             # Documentation structure guide
│   └── guide_aws_builder_account.pdf       # Cloud environment notes
│
├── data/                              # Dataset management
│   ├── raw/ -> ../student_resource/dataset/train  # Symlink to raw training TSVs (no duplication)
│   ├── processed/                          # Benchmark pairs (50k) & engineered feature matrix
│   └── archives/                           # Original zip archives (student_resource.zip)
│
├── student_resource/                  # Official dataset bundle
│   ├── dataset/train/                      # Raw training TSVs (source1, source2, source3, ground_truth)
│   ├── dataset/test/                       # Test sets
│   └── utils/validate_submission.py        # Submission validation utility
│
├── result/                            # Historical benchmark prediction outputs
│   └── matching_results.tsv                # Full predictions output (1.73M rows)
│
├── src/                               # Modular Python source code (Phase-wise implementation)
├── notebooks/                         # Numbered Jupyter notebooks (EDA, Training, Evaluation)
└── results/                           # Experimental outputs (plots, tables, ROC curves, confusion matrices)
    ├── plots/
    └── tables/
```

---

## 🎯 Project Overview & Objective

This project investigates **Business Entity Resolution (Record Linkage)** across heterogeneous, unstandardized business registries. Given a deduplicated master registry (Source 1) and noisy, fragmented records (Sources 2 and 3) featuring cross-script text (Devanagari/Latin), domain names, missing fields, and varied address conventions, the objective is to classify candidate pairs as matching (`1`) or non-matching (`0`).

We conduct an empirical comparative study across core machine learning paradigms taught in **CSE2032**:
1. **Linear & Regularized:** Logistic Regression (L1 Lasso vs. L2 Ridge)
2. **Instance-Based:** K-Nearest Neighbors (KNN)
3. **Margin-Based:** Support Vector Machines (Linear vs. RBF Kernels)
4. **Tree-Based:** Decision Trees (with depth pruning & split criteria analysis)
5. **Ensemble (Bagging):** Random Forest Classifier
6. **Ensemble (Boosting):** Gradient Boosting / LightGBM

Performance is evaluated across Precision, Recall, Macro F₁, **Macro F₀.₅** (penalizing false merges twice as heavily), and AUC-ROC curves using Stratified 5-Fold Cross-Validation.
