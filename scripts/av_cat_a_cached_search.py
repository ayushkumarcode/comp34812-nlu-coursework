"""AV Cat A — Comprehensive model search using cached features.

Loads pre-cached numpy feature arrays (much faster than
re-extracting). Tries many classifier configs.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import save_predictions
from src.scorer import compute_all_metrics, print_metrics

print("=== AV Cat A: Cached Feature Search ===\n")

cache = PROJECT_ROOT / 'models'
try:
    X_tr = np.load(cache / 'av_features_train.npy')
    X_dv = np.load(cache / 'av_features_dev.npy')
    y_tr = np.load(cache / 'av_labels_train.npy')
    y_dv = np.load(cache / 'av_labels_dev.npy')
    print(f"Loaded cached features: {X_tr.shape}")
except FileNotFoundError:
    print("Cached features not found!")
    print("Run cache_av_features.py first")
    sys.exit(1)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_dv = scaler.transform(X_dv)

results = {}

# 1. XGB configurations
configs_xgb = [
    {'n': 1500, 'd': 5, 'lr': 0.02, 'sub': 0.7,
     'col': 0.6, 'ra': 0.1, 'rl': 1.0, 'mcw': 5},
    {'n': 3000, 'd': 3, 'lr': 0.01, 'sub': 0.6,
     'col': 0.5, 'ra': 0.5, 'rl': 2.0, 'mcw': 10},
    {'n': 2000, 'd': 7, 'lr': 0.01, 'sub': 0.8,
     'col': 0.7, 'ra': 0.05, 'rl': 0.5, 'mcw': 3},
]
for i, c in enumerate(configs_xgb):
    name = f"xgb_{i}"
    print(f"\n--- {name} ---")
    m = XGBClassifier(
        n_estimators=c['n'], max_depth=c['d'],
        learning_rate=c['lr'],
        subsample=c['sub'],
        colsample_bytree=c['col'],
        reg_alpha=c['ra'], reg_lambda=c['rl'],
        min_child_weight=c['mcw'],
        eval_metric='logloss',
        random_state=42, n_jobs=-1,
    )
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_dv)[:, 1]
    bf, bt = 0, 0.5
    for t in np.arange(0.35, 0.65, 0.005):
        f1 = f1_score(
            y_dv, (p > t).astype(int),
            average='macro'
        )
        if f1 > bf: bf, bt = f1, t
    print(f"  F1={bf:.4f} (t={bt:.3f})")
    results[name] = (bf, p, bt)

# 2. LGBM configurations
configs_lgbm = [
    {'bt': 'gbdt', 'n': 2000, 'd': 5, 'lr': 0.01,
     'nl': 31, 'mcs': 20},
    {'bt': 'dart', 'n': 1500, 'd': 7, 'lr': 0.02,
     'nl': 63, 'mcs': 10},
    {'bt': 'goss', 'n': 2000, 'd': 5, 'lr': 0.01,
     'nl': 31, 'mcs': 20},
]
for i, c in enumerate(configs_lgbm):
    name = f"lgbm_{c['bt']}_{i}"
    print(f"\n--- {name} ---")
    m = LGBMClassifier(
        boosting_type=c['bt'],
        n_estimators=c['n'],
        max_depth=c['d'],
        learning_rate=c['lr'],
        num_leaves=c['nl'],
        min_child_samples=c['mcs'],
        verbose=-1, random_state=42, n_jobs=-1,
    )
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_dv)[:, 1]
    bf, bt = 0, 0.5
    for t in np.arange(0.35, 0.65, 0.005):
        f1 = f1_score(
            y_dv, (p > t).astype(int),
            average='macro'
        )
        if f1 > bf: bf, bt = f1, t
    print(f"  F1={bf:.4f} (t={bt:.3f})")
    results[name] = (bf, p, bt)

# 3. Extra Trees
print("\n--- ExtraTrees ---")
m_et = ExtraTreesClassifier(
    n_estimators=1000, max_depth=20,
    min_samples_leaf=5,
    random_state=42, n_jobs=-1,
)
m_et.fit(X_tr, y_tr)
p_et = m_et.predict_proba(X_dv)[:, 1]
bf_et, bt_et = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
    f1 = f1_score(
        y_dv, (p_et > t).astype(int),
        average='macro'
    )
    if f1 > bf_et: bf_et, bt_et = f1, t
print(f"  F1={bf_et:.4f} (t={bt_et:.3f})")
results['extra_trees'] = (bf_et, p_et, bt_et)

# 4. Probability ensemble
print("\n=== Prob Ensembles ===")
all_probs = {n: v[1] for n, v in results.items()}
names = list(all_probs.keys())

# Average all
pavg = np.mean(
    [all_probs[n] for n in names], axis=0
)
bf_all, bt_all = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
    f1 = f1_score(
        y_dv, (pavg > t).astype(int),
        average='macro'
    )
    if f1 > bf_all: bf_all, bt_all = f1, t
print(f"  All-avg: F1={bf_all:.4f} (t={bt_all:.3f})")

# Top-3 average
top3 = sorted(results, key=lambda n: results[n][0],
              reverse=True)[:3]
pavg3 = np.mean(
    [all_probs[n] for n in top3], axis=0
)
bf3, bt3 = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
    f1 = f1_score(
        y_dv, (pavg3 > t).astype(int),
        average='macro'
    )
    if f1 > bf3: bf3, bt3 = f1, t
print(f"  Top-3: {top3}: F1={bf3:.4f}")

# Print summary
print("\n=== Summary ===")
for n in sorted(results, key=lambda x: results[x][0],
                reverse=True):
    print(f"  {n:25s}: F1={results[n][0]:.4f}")
print(f"  Ensemble-all: F1={bf_all:.4f}")
print(f"  Ensemble-top3: F1={bf3:.4f}")

# Save best
best_f1 = max(bf_all, bf3,
              max(v[0] for v in results.values()))
if bf3 >= bf_all and bf3 >= max(
    v[0] for v in results.values()
):
    final = (pavg3 > bt3).astype(int)
elif bf_all >= max(v[0] for v in results.values()):
    final = (pavg > bt_all).astype(int)
else:
    bn = max(results, key=lambda n: results[n][0])
    final = (results[bn][1] > results[bn][2]).astype(int)

pred_path = (
    PROJECT_ROOT / 'predictions'
    / 'av_Group_34_A_cached_search.csv'
)
pred_path.parent.mkdir(exist_ok=True)
save_predictions(final, pred_path)

metrics = compute_all_metrics(y_dv, final)
print_metrics(metrics, "AV Cat A Cached Search Best")

for n, bl in [('SVM', 0.5610), ('LSTM', 0.6226),
              ('BERT', 0.7854)]:
    gap = best_f1 - bl
    s = "BEATS" if gap > 0 else "BELOW"
    print(f"  vs {n}: {s} by {gap:+.4f}")
print(f"Current best AV Cat A: 0.7340")
print("Done!")
