#!/usr/bin/env python3
"""
train_rf.py
Train and evaluate a Random Forest Intrusion Detection model using CAN feature data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_features.csv"   # feature file from featurize_can.py
MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.joblib"                # output model file
RESULTS_DIR = PROJECT_ROOT / "results"                                   # metrics + plots
RANDOM_SEED = 42
N_ESTIMATORS = 300
MAX_DEPTH = None
TEST_SIZE = 0.2
# ----------------------------

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Load and prepare data ---
print(f"[+] Loading {DATA_PATH}")
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Feature file not found at {DATA_PATH}. Ensure feature extraction has been run "
        "and the script is executed from within the project directory."
    )

df = pd.read_csv(DATA_PATH)
df = df.fillna(0)

# Remove non-feature columns
non_features = ['win', 'start_time', 'end_time', 'file', 'label']
features = [c for c in df.columns if c not in non_features and df[c].dtype != 'O']

X = df[features].values
y = df['label'].values

unique_classes = np.unique(y)
if unique_classes.size < 2:
    print("[!] Detected single-class dataset; skipping classifier training and computing variance-based feature ranking instead.")

    feature_stats = df[features].agg(['mean', 'std', 'var']).T
    feature_stats.index.name = 'feature'
    feature_stats = feature_stats.rename(columns={'std': 'std_dev', 'var': 'variance'})
    feature_stats = feature_stats.sort_values('variance', ascending=False)

    stats_path = RESULTS_DIR / "baseline_feature_stats.csv"
    feature_stats.to_csv(stats_path)

    top_variance = feature_stats.head(10)
    plt.barh(top_variance.index[::-1], top_variance['variance'][::-1])
    plt.title("Top 10 Features by Variance (Baseline)")
    plt.xlabel("Variance")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "baseline_feature_variance.png", dpi=150)
    plt.close()

    print(f"[✓] Baseline feature statistics saved to {stats_path}")
    print(f"[✓] Variance plot saved to {RESULTS_DIR / 'baseline_feature_variance.png'}")
    raise SystemExit(0)

# Train/Val/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

# --- Normalize features ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Train model ---
print("[+] Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf.fit(X_train, y_train)
dump((rf, scaler), MODEL_PATH)
print(f"[✓] Model saved to {MODEL_PATH}")

# --- Evaluation ---
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1] if hasattr(rf, "predict_proba") else y_pred

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

metrics = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1": f1,
    "ROC_AUC": roc_auc
}
print("\n--- Evaluation Metrics ---")
for k,v in metrics.items():
    print(f"{k:10s}: {v:.4f}")

# --- Save metrics ---
pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "rf_metrics.csv", index=False)

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal","Attack"])
disp.plot(cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.savefig(RESULTS_DIR / "rf_confusion_matrix.png", dpi=150)
plt.close()

# --- Feature Importance ---
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = np.array(features)[indices][:10]
plt.barh(top_features[::-1], importances[indices][:10][::-1])
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "rf_feature_importance.png", dpi=150)
plt.close()

print(f"[✓] Metrics and plots saved in {RESULTS_DIR}")
