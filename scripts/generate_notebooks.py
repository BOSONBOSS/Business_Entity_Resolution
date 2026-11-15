"""
Jupyter Notebook Generator
--------------------------
Generates 6 numbered, fully documented notebooks conforming to the CSE2032 syllabus:
  1. notebooks/01_EDA.ipynb
  2. notebooks/02_Preprocessing.ipynb
  3. notebooks/03_Feature_Engineering.ipynb
  4. notebooks/04_Model_Comparison.ipynb
  5. notebooks/05_Hyperparameter_Tuning.ipynb
  6. notebooks/06_Ensemble_Analysis.ipynb
"""

import os
import nbformat as nbf

NOTEBOOKS_DIR = "notebooks"

def create_notebook(filename, cells):
    nb = nbf.v4.new_notebook()
    nb_cells = []
    for ctype, content in cells:
        if ctype == "markdown":
            nb_cells.append(nbf.v4.new_markdown_cell(content))
        elif ctype == "code":
            nb_cells.append(nbf.v4.new_code_cell(content))
    nb['cells'] = nb_cells
    out_path = os.path.join(NOTEBOOKS_DIR, filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"✅ Generated Notebook: {out_path}")

def generate_all():
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

    # 1. 01_EDA.ipynb
    create_notebook("01_EDA.ipynb", [
        ("markdown", "# 01 — Exploratory Data Analysis (EDA)\n## Business Entity Resolution across Heterogeneous Registries\n**Course:** CSE2032 — Machine Learning | **Dataset:** 26.4M Records Benchmark"),
        ("code", "import os\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nsns.set_theme(style='whitegrid')\n%matplotlib inline"),
        ("markdown", "### 1. Ingesting Benchmark Pairs (50,000 Curated Pairs)"),
        ("code", "benchmark_df = pd.read_csv('../data/processed/benchmark_pairs.csv')\nprint(f'Benchmark Shape: {benchmark_df.shape}')\nbenchmark_df.head(5)"),
        ("markdown", "### 2. Class Balance & Difficulty Strata"),
        ("code", "print('Label Breakdown:')\nprint(benchmark_df['label'].value_counts(normalize=True))\nprint('\\nPair Difficulty Breakdown:')\nprint(benchmark_df['pair_type'].value_counts())"),
        ("markdown", "### 3. Country Distributions across Sources"),
        ("code", "fig, ax = plt.subplots(figsize=(7, 4))\nsns.countplot(data=benchmark_df, x='s1_country', hue='label', palette='Blues', ax=ax)\nax.set_title('Country Distribution by Ground Truth Match Label')\nplt.show()"),
        ("markdown", "### 4. Bimodal Name Similarity & Address Fallback\nVisualizing the distribution of string similarity on true matches vs. hard negatives."),
        ("code", "from src.visualize import plot_eda_distributions\nplot_eda_distributions(benchmark_csv='../data/processed/benchmark_pairs.csv', feature_csv='../data/processed/feature_matrix.csv')\nfrom IPython.display import Image\nImage(filename='../results/plots/eda_distributions.png')")
    ])

    # 2. 02_Preprocessing.ipynb
    create_notebook("02_Preprocessing.ipynb", [
        ("markdown", "# 02 — Text Normalization & Preprocessing Pipeline\n## Reversing Real-World Corruptions in Business Names & Addresses\n**Module:** `src/preprocess.py`"),
        ("code", "import sys, os\nsys.path.insert(0, '..')\nfrom src.preprocess import (\n    preprocess_business_name,\n    preprocess_address,\n    strip_accents,\n    is_non_latin\n)"),
        ("markdown", "### 1. Reversing Latin Accents without Corrupting Indic Scripts"),
        ("code", "cases = [\n    'Moore Bítwise Inc',\n    'B+ Rétail Incorporated',\n    'राम मार्केटिंग प्राइवेट लिमिटेड'\n]\nfor c in cases:\n    print(f'Raw: {c:35s} -> Preprocessed: {preprocess_business_name(c)}')"),
        ("markdown", "### 2. Social Handles & Website Domain Removal"),
        ("code", "web_cases = ['@primemoney', 'moorebitwise.com', 'dhaninvestments.in']\nfor c in web_cases:\n    print(f'Raw: {c:20s} -> Preprocessed: {preprocess_business_name(c)}')"),
        ("markdown", "### 3. Address Zero-Padding & Road Standardization"),
        ("code", "addr_cases = [\n    '0017560 ELLIS ROAD',\n    '0337 Oakland Ave',\n    'PHOENIX, AZ, 1712 MONTEBELLO AVE'\n]\nfor a in addr_cases:\n    print(f'Raw: {a:35s} -> Preprocessed: {preprocess_address(a)}')")
    ])

    # 3. 03_Feature_Engineering.ipynb
    create_notebook("03_Feature_Engineering.ipynb", [
        ("markdown", "# 03 — Feature Engineering & Pairwise Metric Construction\n## Transforming String Pairs into a 15-Dimensional Feature Matrix $\\mathbf{x} \\in \\mathbb{R}^{15}$"),
        ("code", "import pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\nfeat_df = pd.read_csv('../data/processed/feature_matrix.csv')\nprint(f'Feature Matrix Loaded: {feat_df.shape}')\nfeat_df.head(4)"),
        ("markdown", "### 1. Feature Correlation Matrix"),
        ("code", "from src.features import FEATURE_COLUMNS\nplt.figure(figsize=(12, 10))\ncorr = feat_df[FEATURE_COLUMNS].corr()\nsns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)\nplt.title('Pearson Correlation Heatmap of 15 Pairwise Features')\nplt.tight_layout()\nplt.show()"),
        ("markdown", "### 2. Feature Means by Class (Match vs Non-Match)"),
        ("code", "feat_df.groupby('label')[FEATURE_COLUMNS].mean().T")
    ])

    # 4. 04_Model_Comparison.ipynb
    create_notebook("04_Model_Comparison.ipynb", [
        ("markdown", "# 04 — Model Benchmarking & Stratified 5-Fold Cross-Validation\n## Comparative Study across 6 Machine Learning Paradigms (CSE2032)"),
        ("code", "import pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import Image, display\n\nmetrics_df = pd.read_csv('../results/tables/model_comparison_metrics.csv')\nmetrics_df[['Model', 'Accuracy_Mean', 'Precision_Mean', 'Recall_Mean', 'Macro_F1_Mean', 'Macro_F0.5_Mean', 'AUC_ROC_Mean']]"),
        ("markdown", "### 1. Model Leaderboard Visualized"),
        ("code", "plt.figure(figsize=(10, 5))\nsns.barplot(data=metrics_df, x='Macro_F0.5_Mean', y='Model', palette='Blues_r')\nplt.title('Model Comparison Leaderboard: Ranked by Macro F0.5')\nplt.xlabel('Macro F0.5 Score (2x Penalty on False Merges)')\nplt.show()"),
        ("markdown", "### 2. Multi-Model ROC Curves & Precision-Recall Curves"),
        ("code", "display(Image(filename='../results/plots/roc_curves_comparison.png'))\ndisplay(Image(filename='../results/plots/pr_curves_comparison.png'))"),
        ("markdown", "### 3. Confusion Matrix Grid"),
        ("code", "display(Image(filename='../results/plots/confusion_matrices_grid.png'))")
    ])

    # 5. 05_Hyperparameter_Tuning.ipynb
    create_notebook("05_Hyperparameter_Tuning.ipynb", [
        ("markdown", "# 05 — Hyperparameter Optimization & Sensitivity Analysis\n## Tuning Tree Depth, Regularization, and Learning Rates"),
        ("code", "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.model_selection import GridSearchCV\nfrom sklearn.tree import DecisionTreeClassifier\nfrom src.features import FEATURE_COLUMNS\n\ndf = pd.read_csv('../data/processed/feature_matrix.csv')\nX, y = df[FEATURE_COLUMNS], df['label']"),
        ("markdown", "### 1. Decision Tree Depth Sensitivity Curve"),
        ("code", "depths = list(range(2, 20, 2))\ntrain_scores, val_scores = [], []\n\nfor d in depths:\n    dt = DecisionTreeClassifier(max_depth=d, random_state=42)\n    dt.fit(X[:35000], y[:35000])\n    train_scores.append(dt.score(X[:35000], y[:35000]))\n    val_scores.append(dt.score(X[35000:], y[35000:]))\n\nplt.figure(figsize=(8, 5))\nplt.plot(depths, train_scores, 'o-', label='Training Accuracy')\nplt.plot(depths, val_scores, 's--', label='Validation Accuracy')\nplt.xlabel('Tree Max Depth')\nplt.ylabel('Accuracy')\nplt.title('Decision Tree Overfitting & Depth Pruning Analysis')\nplt.legend()\nplt.show()")
    ])

    # 6. 06_Ensemble_Analysis.ipynb
    create_notebook("06_Ensemble_Analysis.ipynb", [
        ("markdown", "# 06 — Ensemble Analysis, SHAP Interpretability & Error Audit\n## Explaining Model Decisions and Auditing Failure Modes"),
        ("code", "import pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import Image, display\nfrom src.features import FEATURE_COLUMNS"),
        ("markdown", "### 1. LightGBM Feature Importances"),
        ("code", "display(Image(filename='../results/plots/feature_importance_lgbm.png'))"),
        ("markdown", "### 2. Qualitative Error Audit: Inspecting Top False Positives & Negatives"),
        ("code", "oof = pd.read_csv('../results/tables/oof_predictions.csv')\nb_df = pd.read_csv('../data/processed/benchmark_pairs.csv')\n\nmerged = pd.concat([b_df, oof[['LightGBM_prob', 'LightGBM_pred']]], axis=1)\n\n# False Positives: Model predicted match (high prob), but true label is 0\nfp = merged[(merged['label'] == 0) & (merged['LightGBM_pred'] == 1)].sort_values(by='LightGBM_prob', ascending=False)\nprint(f'Total False Positives: {len(fp):,}')\nfp[['s1_name', 's1_address', 'probe_name', 'probe_address', 'pair_type', 'LightGBM_prob']].head(5)")
    ])

if __name__ == "__main__":
    generate_all()

