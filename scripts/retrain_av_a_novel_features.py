"""AV Cat A retrain with 5 novel feature groups.

Re-extracts features (now ~715+ with FFT spectral, Zipf-Mandelbrot,
Benford's law, Hurst exponent, and Cosine Delta), then trains
multiple GBM configs and compares against old cached baseline.
"""
import sys
import time
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (
    load_av_data, load_solution_labels, save_predictions
)
from src.av_pipeline import AVFeatureExtractor
from src.scorer import compute_all_metrics, print_metrics

print("=" * 60)
print("AV Cat A: Retrain with 5 Novel Feature Groups")
print("=" * 60)

# --- load data ---
t0 = time.time()
train_df = load_av_data(split='train')
dev_df = load_av_data(split='dev')
y_train = train_df['label'].values
y_dev = np.array(load_solution_labels(task='av'))
print(f"Data loaded in {time.time()-t0:.1f}s")
print(f"Train: {len(train_df)}, Dev: {len(dev_df)}")

# --- extract features with new groups ---
print("\nExtracting features (with novel groups)...")
t0 = time.time()
ext = AVFeatureExtractor(use_spacy=True, n_svd_components=100)
ext.fit(train_df)
X_train, fnames = ext.transform(train_df)
X_dev, _ = ext.transform(dev_df)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")
print(f"Feature matrix: {X_train.shape[1]} features")

# show novel feature names
novel_prefixes = ['fft_', 'zipf_', 'benford_', 'hurst_', 'cosine_delta']
novel_feats = [f for f in fnames if any(f.startswith(p) or p in f for p in novel_prefixes)]
print(f"\nNovel features ({len(novel_feats)}):")
for f in sorted(novel_feats):
    print(f"  {f}")

# --- cache new features ---
cache_dir = PROJECT_ROOT / 'models'
cache_dir.mkdir(exist_ok=True)
np.save(cache_dir / 'av_features_train_v2.npy', X_train)
np.save(cache_dir / 'av_features_dev_v2.npy', X_dev)
with open(cache_dir / 'av_feature_names_v2.txt', 'w') as f:
    for name in fnames:
        f.write(f"{name}\n")
print(f"\nCached to {cache_dir} (v2)")

# --- scale ---
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_dv = scaler.transform(X_dev)

# --- old baseline (from cached features if available) ---
old_f1 = None
try:
    X_old_tr = np.load(cache_dir / 'av_features_train.npy')
    X_old_dv = np.load(cache_dir / 'av_features_dev.npy')
    print(f"\nOld features: {X_old_tr.shape[1]} dims")
    sc_old = StandardScaler()
    X_old_tr = sc_old.fit_transform(X_old_tr)
    X_old_dv = sc_old.transform(X_old_dv)

    m_old = LGBMClassifier(
        boosting_type='dart', n_estimators=2000,
        max_depth=7, learning_rate=0.02,
        num_leaves=63, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.7,
        verbose=-1, random_state=42, n_jobs=-1,
    )
    m_old.fit(X_old_tr, y_train)
    p_old = m_old.predict_proba(X_old_dv)[:, 1]
    old_f1 = 0
    for t in np.arange(0.35, 0.65, 0.005):
        f1 = f1_score(y_dev, (p_old > t).astype(int), average='macro')
        if f1 > old_f1:
            old_f1 = f1
    print(f"Old features best F1: {old_f1:.4f}")
except FileNotFoundError:
    print("\nNo old cached features found, skipping baseline comparison")

# --- train new models ---
results = {}

# LGBM dart
print("\n--- LGBM dart (new features) ---")
m1 = LGBMClassifier(
    boosting_type='dart', n_estimators=2000,
    max_depth=7, learning_rate=0.02,
    num_leaves=63, min_child_samples=10,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0,
    drop_rate=0.1,
    verbose=-1, random_state=42, n_jobs=-1,
)
m1.fit(X_tr, y_train)
p1 = m1.predict_proba(X_dv)[:, 1]
bf1, bt1 = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
    f1 = f1_score(y_dev, (p1 > t).astype(int), average='macro')
    if f1 > bf1: bf1, bt1 = f1, t
print(f"  F1={bf1:.4f} (t={bt1:.3f})")
results['lgbm_dart'] = (bf1, p1, bt1)

# LGBM goss
print("\n--- LGBM goss (new features) ---")
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
    f1 = f1_score(y_dev, (p2 > t).astype(int), average='macro')
    if f1 > bf2: bf2, bt2 = f1, t
print(f"  F1={bf2:.4f} (t={bt2:.3f})")
results['lgbm_goss'] = (bf2, p2, bt2)

# XGB shallow + regularized
print("\n--- XGB shallow (new features) ---")
m3 = XGBClassifier(
    n_estimators=3000, max_depth=3,
    learning_rate=0.01,
    subsample=0.6, colsample_bytree=0.5,
    reg_alpha=1.0, reg_lambda=3.0,
    min_child_weight=10, gamma=0.2,
    eval_metric='logloss',
    random_state=42, n_jobs=-1,
)
m3.fit(X_tr, y_train)
p3 = m3.predict_proba(X_dv)[:, 1]
bf3, bt3 = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
    f1 = f1_score(y_dev, (p3 > t).astype(int), average='macro')
    if f1 > bf3: bf3, bt3 = f1, t
print(f"  F1={bf3:.4f} (t={bt3:.3f})")
results['xgb_shallow'] = (bf3, p3, bt3)

# LGBM gbdt (vanilla)
print("\n--- LGBM gbdt (new features) ---")
m4 = LGBMClassifier(
    n_estimators=2000, max_depth=5,
    learning_rate=0.01, num_leaves=31,
    min_child_samples=20,
    subsample=0.7, colsample_bytree=0.6,
    verbose=-1, random_state=42, n_jobs=-1,
)
m4.fit(X_tr, y_train)
p4 = m4.predict_proba(X_dv)[:, 1]
bf4, bt4 = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
    f1 = f1_score(y_dev, (p4 > t).astype(int), average='macro')
    if f1 > bf4: bf4, bt4 = f1, t
print(f"  F1={bf4:.4f} (t={bt4:.3f})")
results['lgbm_gbdt'] = (bf4, p4, bt4)

# --- ensemble ---
print("\n--- Ensemble (avg probs) ---")
all_probs = [v[1] for v in results.values()]
pavg = np.mean(all_probs, axis=0)
bf_e, bt_e = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
    f1 = f1_score(y_dev, (pavg > t).astype(int), average='macro')
    if f1 > bf_e: bf_e, bt_e = f1, t
print(f"  F1={bf_e:.4f} (t={bt_e:.3f})")
results['ensemble_4'] = (bf_e, pavg, bt_e)

# top-2 ensemble
top2 = sorted(results, key=lambda n: results[n][0], reverse=True)[:2]
pavg2 = np.mean([results[n][1] for n in top2], axis=0)
bf_e2, bt_e2 = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
    f1 = f1_score(y_dev, (pavg2 > t).astype(int), average='macro')
    if f1 > bf_e2: bf_e2, bt_e2 = f1, t
print(f"  Top-2 ({top2}): F1={bf_e2:.4f} (t={bt_e2:.3f})")

# --- summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name in sorted(results, key=lambda n: results[n][0], reverse=True):
    f1_val = results[name][0]
    marker = ""
    if old_f1 is not None:
        delta = f1_val - old_f1
        marker = f" ({delta:+.4f} vs old)"
    print(f"  {name:20s}: F1={f1_val:.4f}{marker}")
print(f"  top2_ensemble     : F1={bf_e2:.4f}")

best_name = max(results, key=lambda n: results[n][0])
best_f1 = max(results[best_name][0], bf_e2)
print(f"\nBest new F1: {best_f1:.4f}")
if old_f1 is not None:
    delta = best_f1 - old_f1
    print(f"Old F1:      {old_f1:.4f}")
    print(f"Improvement: {delta:+.4f}")

# baselines
for n, bl in [('SVM', 0.5610), ('LSTM', 0.6226), ('BERT', 0.7854)]:
    gap = best_f1 - bl
    s = "BEATS" if gap > 0 else "BELOW"
    print(f"  vs {n}: {s} by {gap:+.4f}")

# save best predictions
if bf_e2 >= results[best_name][0]:
    final = (pavg2 > bt_e2).astype(int)
else:
    bn = best_name
    final = (results[bn][1] > results[bn][2]).astype(int)

pred_path = PROJECT_ROOT / 'predictions' / 'av_Group_34_A_novel.csv'
pred_path.parent.mkdir(exist_ok=True)
save_predictions(final, pred_path)

metrics = compute_all_metrics(y_dev, final)
print_metrics(metrics, "AV Cat A Novel Features")
print("Done!")
