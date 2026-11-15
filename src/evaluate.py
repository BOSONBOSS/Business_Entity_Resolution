"""
Comprehensive Model Evaluation & 5-Fold Cross-Validation Engine
---------------------------------------------------------------
Executes Stratified 5-Fold Cross-Validation across all 6 ML paradigms.
Features are standardized strictly per fold using StandardScaler.

Metrics Computed:
  - Accuracy
  - Precision (Macro & Binary)
  - Recall (Macro & Binary)
  - Macro F1
  - Macro F0.5 (Primary Metric: 2x penalty on false merges)
  - AUC-ROC

Outputs:
  - results/tables/model_comparison_metrics.csv
  - results/tables/oof_predictions.csv (Out-of-fold predictions for visualization)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import get_models
from src.features import FEATURE_COLUMNS

RESULTS_DIR = "results"
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FEATURE_MATRIX_PATH = "data/processed/feature_matrix.csv"


def run_cross_validation(n_splits=5, random_state=42):
    start_time = time.time()
    os.makedirs(TABLES_DIR, exist_ok=True)
    
    print("=" * 80)
    print("🚀 STRATIFIED 5-FOLD CROSS-VALIDATION EXPERIMENT")
    print("=" * 80)

    print(f"Loading feature matrix: {FEATURE_MATRIX_PATH}...")
    df = pd.read_csv(FEATURE_MATRIX_PATH)
    
    X = df[FEATURE_COLUMNS].values
    y = df['label'].values
    print(f"Dataset shape: X={X.shape}, y={y.shape} (Positives: {(y==1).sum():,}, Negatives: {(y==0).sum():,})")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models = get_models(random_state=random_state)

    results_summary = []
    oof_predictions = {'label': y}

    for model_name, model in models.items():
        print(f"\n--- Training & Evaluating: {model_name} ---")
        m_start = time.time()
        
        fold_accs = []
        fold_precs = []
        fold_recalls = []
        fold_f1s = []
        fold_f05s = []
        fold_aucs = []

        oof_probs = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit scaler strictly on training fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Fit model
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_val_scaled)
            
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_val_scaled)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_val_scaled)
            else:
                y_prob = y_pred.astype(float)

            oof_probs[val_idx] = y_prob

            # Metric computations
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_val, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
            f05 = fbeta_score(y_val, y_pred, beta=0.5, average='macro', zero_division=0)
            auc = roc_auc_score(y_val, y_prob)

            fold_accs.append(acc)
            fold_precs.append(prec)
            fold_recalls.append(rec)
            fold_f1s.append(f1)
            fold_f05s.append(f05)
            fold_aucs.append(auc)

        oof_predictions[f"{model_name}_prob"] = oof_probs
        oof_predictions[f"{model_name}_pred"] = (oof_probs >= 0.5).astype(int)

        m_elapsed = time.time() - m_start
        mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
        mean_prec, std_prec = np.mean(fold_precs), np.std(fold_precs)
        mean_rec, std_rec = np.mean(fold_recalls), np.std(fold_recalls)
        mean_f1, std_f1 = np.mean(fold_f1s), np.std(fold_f1s)
        mean_f05, std_f05 = np.mean(fold_f05s), np.std(fold_f05s)
        mean_auc, std_auc = np.mean(fold_aucs), np.std(fold_aucs)

        print(f"  Completed in {m_elapsed:.2f}s | Macro F0.5: {mean_f05:.4f} (+/- {std_f05:.4f}) | AUC: {mean_auc:.4f}")

        results_summary.append({
            'Model': model_name,
            'Accuracy_Mean': mean_acc,
            'Accuracy_Std': std_acc,
            'Precision_Mean': mean_prec,
            'Precision_Std': std_prec,
            'Recall_Mean': mean_rec,
            'Recall_Std': std_rec,
            'Macro_F1_Mean': mean_f1,
            'Macro_F1_Std': std_f1,
            'Macro_F0.5_Mean': mean_f05,
            'Macro_F0.5_Std': std_f05,
            'AUC_ROC_Mean': mean_auc,
            'AUC_ROC_Std': std_auc,
            'Train_Time_Sec': m_elapsed
        })

    # Save summary table
    summary_df = pd.DataFrame(results_summary).sort_values(by='Macro_F0.5_Mean', ascending=False)
    metrics_csv_path = os.path.join(TABLES_DIR, "model_comparison_metrics.csv")
    summary_df.to_csv(metrics_csv_path, index=False)

    # Save OOF predictions
    oof_df = pd.DataFrame(oof_predictions)
    oof_path = os.path.join(TABLES_DIR, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✅ 5-FOLD CROSS-VALIDATION COMPLETE in {total_elapsed:.2f} seconds!")
    print(f"   Saved Metrics Table to: {metrics_csv_path}")
    print(f"   Saved OOF Predictions to: {oof_path}")
    print("=" * 80)

    # Print Clean Leaderboard
    leaderboard = summary_df[['Model', 'Accuracy_Mean', 'Precision_Mean', 'Recall_Mean', 'Macro_F1_Mean', 'Macro_F0.5_Mean', 'AUC_ROC_Mean']].copy()
    leaderboard.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'Macro F1', 'Macro F0.5', 'AUC-ROC']
    print("\n🏆 FINAL MODEL COMPARISON LEADERBOARD (Ranked by Macro F0.5):")
    print(leaderboard.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 80)

    return summary_df, oof_df


if __name__ == "__main__":
    run_cross_validation()

