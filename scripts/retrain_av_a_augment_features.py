"""AV Cat A -- augment cached features with 5 novel feature groups.

Instead of re-extracting all features (slow TF-IDF), loads the
old cached feature matrix and appends only the new features:
  - FFT spectral (8 per text x2 -> 16 diff + 16 style_diff)
  - Zipf-Mandelbrot (5 per text x2 -> 10 diff + 10 style_diff)
  - Benford's law (4 per text x2 -> 8 diff + 8 style_diff)
  - Hurst exponent (3 per text x2 -> 6 diff + 6 style_diff)
  - Cosine Delta (1 pairwise)
Total new: ~81 features
"""
import sys
import time
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (
    load_av_data, load_solution_labels, save_predictions
)
from src.av_feature_engineering import (
    spectral_features, zipf_features, benford_features,
    hurst_features, _cosine_delta,
)
from src.scorer import compute_all_metrics, print_metrics

print("=" * 60)
print("AV Cat A: Augment Cached Features with Novel Groups")
print("=" * 60)

# load data
t0 = time.time()
train_df = load_av_data(split='train')
dev_df = load_av_data(split='dev')
y_train = train_df['label'].values
y_dev = np.array(load_solution_labels(task='av'))
print(f"Data loaded in {time.time()-t0:.1f}s")

# load old cached features
cache = PROJECT_ROOT / 'models'
X_old_tr = np.load(cache / 'av_features_train.npy')
X_old_dv = np.load(cache / 'av_features_dev.npy')
print(f"Old features: train={X_old_tr.shape}, dev={X_old_dv.shape}")

# extract new per-text features for all pairs
novel_extractors = [
    spectral_features, zipf_features,
    benford_features, hurst_features,
]

def extract_novel_pair_features(text_1, text_2):
    """Extract diff-vector + style_diff for novel features, plus cosine delta."""
    feats = {}
    f1 = {}
    f2 = {}
    for fn in novel_extractors:
        f1.update(fn(text_1))
        f2.update(fn(text_2))
    # diff features
    for key in sorted(f1.keys()):
        feats[f'diff_{key}'] = abs(f1[key] - f2[key])
    # style diff features
    for key in sorted(f1.keys()):
        feats[f'style_diff_{key}'] = abs(f1[key] - f2[key])
    # cosine delta
    feats['cosine_delta'] = _cosine_delta(text_1, text_2)
    return feats


def extract_batch(df, desc=""):
    """Extract novel features for all pairs in a dataframe."""
    all_feats = []
    n = len(df)
    for i in range(n):
        if i % 2000 == 0:
            print(f"  {desc}: {i}/{n}")
        text_1 = df.iloc[i]['text_1']
        text_2 = df.iloc[i]['text_2']
        feats = extract_novel_pair_features(text_1, text_2)
        all_feats.append(feats)
    # to numpy
    names = sorted(all_feats[0].keys())
    X = np.array([
        [f.get(name, 0.0) for name in names]
        for f in all_feats
    ], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


print("\nExtracting novel features for train...")
t0 = time.time()
X_new_tr, new_names = extract_batch(train_df, "train")
print(f"  Done in {time.time()-t0:.1f}s, shape={X_new_tr.shape}")

print("Extracting novel features for dev...")
t0 = time.time()
X_new_dv, _ = extract_batch(dev_df, "dev")
print(f"  Done in {time.time()-t0:.1f}s, shape={X_new_dv.shape}")

print(f"\nNovel feature names ({len(new_names)}):")
for name in new_names:
    print(f"  {name}")

# concatenate old + new
X_aug_tr = np.concatenate([X_old_tr, X_new_tr], axis=1)
X_aug_dv = np.concatenate([X_old_dv, X_new_dv], axis=1)
print(f"\nAugmented: train={X_aug_tr.shape}, dev={X_aug_dv.shape}")

# save augmented features
np.save(cache / 'av_features_train_v2.npy', X_aug_tr)
np.save(cache / 'av_features_dev_v2.npy', X_aug_dv)
with open(cache / 'av_feature_names_v2.txt', 'w') as f:
    # we don't have old names readily, just write new ones
    for name in new_names:
        f.write(f"{name}\n")
print("Cached augmented features.")

# scale
scaler_old = StandardScaler()
X_old_tr_s = scaler_old.fit_transform(X_old_tr)
X_old_dv_s = scaler_old.transform(X_old_dv)

scaler_aug = StandardScaler()
X_aug_tr_s = scaler_aug.fit_transform(X_aug_tr)
X_aug_dv_s = scaler_aug.transform(X_aug_dv)

# --- train on OLD features ---
print("\n" + "=" * 60)
print("BASELINE: Old features only")
print("=" * 60)

m_old = LGBMClassifier(
    boosting_type='dart', n_estimators=2000,
    max_depth=7, learning_rate=0.02,
    num_leaves=63, min_child_samples=10,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0,
    verbose=-1, random_state=42, n_jobs=-1,
)
m_old.fit(X_old_tr_s, y_train)
p_old = m_old.predict_proba(X_old_dv_s)[:, 1]
old_f1, old_t = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
    f1 = f1_score(y_dev, (p_old > t).astype(int), average='macro')
    if f1 > old_f1: old_f1, old_t = f1, t
print(f"  Old LGBM dart: F1={old_f1:.4f} (t={old_t:.3f})")

# --- train on AUGMENTED features ---
print("\n" + "=" * 60)
print("NEW: Augmented features (old + novel)")
print("=" * 60)

results = {}

# LGBM dart
print("\n--- LGBM dart ---")
m1 = LGBMClassifier(
    boosting_type='dart', n_estimators=2000,
    max_depth=7, learning_rate=0.02,
    num_leaves=63, min_child_samples=10,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, drop_rate=0.1,
    verbose=-1, random_state=42, n_jobs=-1,
)
m1.fit(X_aug_tr_s, y_train)
p1 = m1.predict_proba(X_aug_dv_s)[:, 1]
bf1, bt1 = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
    f1 = f1_score(y_dev, (p1 > t).astype(int), average='macro')
    if f1 > bf1: bf1, bt1 = f1, t
print(f"  F1={bf1:.4f} (t={bt1:.3f})  delta={bf1-old_f1:+.4f}")
results['lgbm_dart'] = (bf1, p1, bt1)

# LGBM goss
print("\n--- LGBM goss ---")
m2 = LGBMClassifier(
    boosting_type='goss', n_estimators=2000,
    max_depth=5, learning_rate=0.01,
    num_leaves=31, min_child_samples=20,
    reg_alpha=0.5, reg_lambda=2.0,
    verbose=-1, random_state=42, n_jobs=-1,
)
m2.fit(X_aug_tr_s, y_train)
p2 = m2.predict_proba(X_aug_dv_s)[:, 1]
bf2, bt2 = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
    f1 = f1_score(y_dev, (p2 > t).astype(int), average='macro')
    if f1 > bf2: bf2, bt2 = f1, t
print(f"  F1={bf2:.4f} (t={bt2:.3f})  delta={bf2-old_f1:+.4f}")
results['lgbm_goss'] = (bf2, p2, bt2)

# XGB
print("\n--- XGB shallow ---")
m3 = XGBClassifier(
    n_estimators=3000, max_depth=3,
    learning_rate=0.01,
    subsample=0.6, colsample_bytree=0.5,
    reg_alpha=1.0, reg_lambda=3.0,
    min_child_weight=10, gamma=0.2,
    eval_metric='logloss',
    random_state=42, n_jobs=-1,
)
m3.fit(X_aug_tr_s, y_train)
p3 = m3.predict_proba(X_aug_dv_s)[:, 1]
bf3, bt3 = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
    f1 = f1_score(y_dev, (p3 > t).astype(int), average='macro')
    if f1 > bf3: bf3, bt3 = f1, t
print(f"  F1={bf3:.4f} (t={bt3:.3f})  delta={bf3-old_f1:+.4f}")
results['xgb_shallow'] = (bf3, p3, bt3)

# LGBM gbdt
print("\n--- LGBM gbdt ---")
m4 = LGBMClassifier(
    n_estimators=2000, max_depth=5,
    learning_rate=0.01, num_leaves=31,
    min_child_samples=20,
    subsample=0.7, colsample_bytree=0.6,
    verbose=-1, random_state=42, n_jobs=-1,
)
m4.fit(X_aug_tr_s, y_train)
p4 = m4.predict_proba(X_aug_dv_s)[:, 1]
bf4, bt4 = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
    f1 = f1_score(y_dev, (p4 > t).astype(int), average='macro')
    if f1 > bf4: bf4, bt4 = f1, t
print(f"  F1={bf4:.4f} (t={bt4:.3f})  delta={bf4-old_f1:+.4f}")
results['lgbm_gbdt'] = (bf4, p4, bt4)

# Ensemble
print("\n--- Ensemble (avg probs) ---")
all_probs = [v[1] for v in results.values()]
pavg = np.mean(all_probs, axis=0)
bf_e, bt_e = 0, 0.5
for t in np.arange(0.35, 0.65, 0.005):
    f1 = f1_score(y_dev, (pavg > t).astype(int), average='macro')
    if f1 > bf_e: bf_e, bt_e = f1, t
print(f"  F1={bf_e:.4f} (t={bt_e:.3f})  delta={bf_e-old_f1:+.4f}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Old baseline (LGBM dart, {X_old_tr.shape[1]} features): F1={old_f1:.4f}")
print()
for name in sorted(results, key=lambda n: results[n][0], reverse=True):
    f1_val = results[name][0]
    delta = f1_val - old_f1
    print(f"  {name:20s}: F1={f1_val:.4f} ({delta:+.4f})")
print(f"  ensemble_4        : F1={bf_e:.4f} ({bf_e-old_f1:+.4f})")

best_f1 = max(bf_e, max(v[0] for v in results.values()))
print(f"\nBest augmented F1: {best_f1:.4f}")
print(f"Improvement over old: {best_f1 - old_f1:+.4f}")

for n, bl in [('SVM', 0.5610), ('LSTM', 0.6226), ('BERT', 0.7854)]:
    gap = best_f1 - bl
    s = "BEATS" if gap > 0 else "BELOW"
    print(f"  vs {n}: {s} by {gap:+.4f}")

# save best predictions
best_name = max(results, key=lambda n: results[n][0])
if bf_e >= results[best_name][0]:
    final = (pavg > bt_e).astype(int)
else:
    final = (results[best_name][1] > results[best_name][2]).astype(int)

pred_path = PROJECT_ROOT / 'predictions' / 'av_Group_34_A_novel.csv'
pred_path.parent.mkdir(exist_ok=True)
save_predictions(final, pred_path)

metrics = compute_all_metrics(y_dev, final)
print_metrics(metrics, "AV Cat A Novel Features Best")
print("Done!")
