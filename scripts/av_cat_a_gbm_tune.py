"""AV Cat A — Gradient boosting with extensive tuning.

Tests multiple configurations:
1. LightGBM with dart boosting
2. XGBoost with gbtree + high regularization
3. CatBoost (if available)
4. Voting ensemble of best models
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import (
    VotingClassifier, GradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (
    load_av_data, load_solution_labels, save_predictions
)
from src.av_pipeline import AVFeatureExtractor
from src.scorer import compute_all_metrics, print_metrics

print("=== AV Cat A: GBM Tuning ===\n")

train_df = load_av_data(split='train')
dev_df = load_av_data(split='dev')
y_train = train_df['label'].values
y_dev = np.array(load_solution_labels(task='av'))

print("Extracting features...")
ext = AVFeatureExtractor(
    use_spacy=True, n_svd_components=100
)
ext.fit(train_df)
X_train, fnames = ext.transform(train_df)
X_dev, _ = ext.transform(dev_df)
print(f"Features: {X_train.shape[1]}")

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_dv = scaler.transform(X_dev)

results = {}

# Config 1: LGBM with dart
print("\n--- LGBM dart ---")
m1 = LGBMClassifier(
    boosting_type='dart', n_estimators=2000,
    max_depth=7, learning_rate=0.02,
    num_leaves=63, min_child_samples=10,
    reg_alpha=0.1, reg_lambda=1.0,
    subsample=0.8, colsample_bytree=0.7,
    drop_rate=0.1,
    verbose=-1, random_state=42, n_jobs=-1,
)
m1.fit(X_tr, y_train)
p1 = m1.predict_proba(X_dv)[:, 1]
bf1, bt1 = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
    f1 = f1_score(y_dev, (p1 > t).astype(int),
                  average='macro')
    if f1 > bf1: bf1, bt1 = f1, t
print(f"  F1={bf1:.4f} (t={bt1:.3f})")
results['lgbm_dart'] = bf1

# Config 2: LGBM with GOSS
print("\n--- LGBM goss ---")
m2 = LGBMClassifier(
    boosting_type='goss', n_estimators=2000,
    max_depth=5, learning_rate=0.01,
    num_leaves=31, min_child_samples=20,
    reg_alpha=0.5, reg_lambda=2.0,
    verbose=-1, random_state=42, n_jobs=-1,
)
m2.fit(X_tr, y_train)
p2 = m2.predict_proba(X_dv)[:, 1]
bf2, bt2 = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
