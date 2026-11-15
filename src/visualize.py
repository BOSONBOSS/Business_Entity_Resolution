"""
Visualization & Plotting Module for Model Analysis and EDA
----------------------------------------------------------
Generates high-resolution (300 DPI) publication-quality figures:
  1. results/plots/roc_curves_comparison.png
  2. results/plots/pr_curves_comparison.png
  3. results/plots/confusion_matrices_grid.png
  4. results/plots/feature_importance_lgbm.png
  5. results/plots/eda_distributions.png

Used directly in the academic report (Final_Project_Report.docx).
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

PLOTS_DIR = "results/plots"
TABLES_DIR = "results/tables"

# Professional styling for academic publication
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})


def plot_eda_distributions(benchmark_csv="data/processed/benchmark_pairs.csv",
                           feature_csv="data/processed/feature_matrix.csv"):
    """Generate 4-panel exploratory data distribution figure."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    b_df = pd.read_csv(benchmark_csv)
    f_df = pd.read_csv(feature_csv)

    # Panel 1: Country Distribution
    ax1 = axes[0, 0]
    country_counts = b_df['s1_country'].value_counts()
    colors = ['#2b5c8f', '#e67e22', '#27ae60']
    ax1.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%',
            startangle=140, colors=colors[:len(country_counts)], explode=[0.05]*len(country_counts))
    ax1.set_title("A. Country Distribution in Benchmark Sample")

    # Panel 2: Pair Difficulty Composition
    ax2 = axes[0, 1]
    p_types = b_df['pair_type'].value_counts()
    sns.barplot(x=p_types.values, y=p_types.index, ax=ax2, palette="Blues_r")
    ax2.set_title("B. Benchmark Pair Type Breakdown (50,000 Pairs)")
    ax2.set_xlabel("Number of Pairs")

    # Panel 3: Name Jaccard Bimodal Distribution (Positives vs Negatives)
    ax3 = axes[1, 0]
    sns.histplot(data=f_df, x='name_token_jaccard', hue='label', bins=30, kde=True,
                 palette={0: '#e74c3c', 1: '#27ae60'}, ax=ax3, stat="density", common_norm=False)
    ax3.set_title("C. Token Jaccard Distribution (Positives vs. Hard Negatives)")
    ax3.set_xlabel("Word-Token Jaccard Similarity")
    ax3.legend(["Match (y=1)", "Non-Match (y=0)"], loc="upper center")

    # Panel 4: Script Mismatch vs Address Fallback
    ax4 = axes[1, 1]
    pos_feats = f_df[f_df['label'] == 1]
    sns.kdeplot(data=pos_feats, x='addr_token_jaccard', hue='is_script_mismatch',
                palette={0.0: '#2980b9', 1.0: '#8e44ad'}, ax=ax4, common_norm=False)
    ax4.set_title("D. Address Jaccard on True Matches (Latin vs. Indic Script)")
    ax4.set_xlabel("Address Token Jaccard Similarity")
    ax4.legend(["Script Mismatch (Indic)", "Standard Latin"], loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "eda_distributions.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved EDA Plot to: {out_path}")


def plot_roc_and_pr_curves(oof_csv="results/tables/oof_predictions.csv"):
    """Generate comparative ROC and Precision-Recall curves across all models."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    oof = pd.read_csv(oof_csv)
    y_true = oof['label'].values

    prob_cols = [c for c in oof.columns if c.endswith('_prob')]
    model_names = [c.replace('_prob', '') for c in prob_cols]

    palette = sns.color_palette("tab10", len(model_names))

    # 1. ROC Curves
    plt.figure(figsize=(9, 7))
    for idx, (m_name, p_col) in enumerate(zip(model_names, prob_cols)):
        y_prob = oof[p_col].values
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        clean_name = m_name.replace('_', ' ')
        plt.plot(fpr, tpr, label=f"{clean_name} (AUC = {roc_auc:.4f})", color=palette[idx], lw=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("Comparative ROC Curves across 6 Machine Learning Paradigms")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(PLOTS_DIR, "roc_curves_comparison.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"✅ Saved ROC Curves to: {roc_path}")

    # 2. Precision-Recall Curves
    plt.figure(figsize=(9, 7))
    for idx, (m_name, p_col) in enumerate(zip(model_names, prob_cols)):
        y_prob = oof[p_col].values
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec, prec)
        clean_name = m_name.replace('_', ' ')
        plt.plot(rec, prec, label=f"{clean_name} (PR-AUC = {pr_auc:.4f})", color=palette[idx], lw=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Focus on High-Precision Entity Linking)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(PLOTS_DIR, "pr_curves_comparison.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"✅ Saved PR Curves to: {pr_path}")


def plot_confusion_matrices(oof_csv="results/tables/oof_predictions.csv"):
    """Plot grid of normalized confusion matrices for each paradigm."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    oof = pd.read_csv(oof_csv)
    y_true = oof['label'].values

    pred_cols = [c for c in oof.columns if c.endswith('_pred')]
    model_names = [c.replace('_pred', '') for c in pred_cols]

    n_models = len(model_names)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = axes.flatten()

    for idx, (m_name, p_col) in enumerate(zip(model_names, pred_cols)):
        y_pred = oof[p_col].values
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        clean_name = m_name.replace('_', ' ')
        ax.set_title(clean_name, fontweight='bold')

    # Hide extra unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrices_grid.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"✅ Saved Confusion Matrix Grid to: {cm_path}")


def plot_feature_importances(feature_csv="data/processed/feature_matrix.csv"):
    """Train LightGBM on full benchmark to extract and plot feature importances."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    from lightgbm import LGBMClassifier
    from src.features import FEATURE_COLUMNS

    df = pd.read_csv(feature_csv)
    X = df[FEATURE_COLUMNS]
    y = df['label']

    lgb = LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1)
    lgb.fit(X, y)

    importances = pd.Series(lgb.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=True)

    plt.figure(figsize=(10, 7))
    importances.plot(kind='barh', color='#2980b9')
    plt.xlabel("Split Feature Importance (Number of Tree Splits)")
    plt.title("LightGBM Feature Importance (Pairwise Entity Resolution)")
    plt.tight_layout()
    fi_path = os.path.join(PLOTS_DIR, "feature_importance_lgbm.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"✅ Saved Feature Importance Plot to: {fi_path}")


def generate_all_visualizations():
    print("=" * 75)
    print("🚀 GENERATING PUBLICATION-QUALITY FIGURES FOR FINAL REPORT")
    print("=" * 75)
    plot_eda_distributions()
    plot_roc_and_pr_curves()
    plot_confusion_matrices()
    plot_feature_importances()
    print("=" * 75)
    print("✅ All visual figures generated successfully!")
    print("=" * 75)


if __name__ == "__main__":
    generate_all_visualizations()
