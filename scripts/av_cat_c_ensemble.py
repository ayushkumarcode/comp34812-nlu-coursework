"""AV Cat C — Ensemble of DeBERTa variants.

Averages prediction probabilities from multiple Cat C runs
and threshold-optimizes for best macro_f1.
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (
    load_solution_labels, save_predictions
)
from src.scorer import compute_all_metrics, print_metrics

y_dev = np.array(load_solution_labels(task='av'))
pred_dir = PROJECT_ROOT / 'predictions'

# Load all available Cat C probability files
prob_files = {
    'rdrop': 'av_cat_c_rdrop_probs.npy',
    'rdrop_v2': 'av_cat_c_rdrop_v2_probs.npy',
    'lr1e5': 'av_cat_c_lr1e5_probs.npy',
    'awp': 'av_cat_c_awp_probs.npy',
    'maxlen256': 'av_cat_c_maxlen256_probs.npy',
    'cosine': 'av_cat_c_cosine_probs.npy',
}

probs = {}
for name, fname in prob_files.items():
    fpath = pred_dir / fname
    if fpath.exists():
        probs[name] = np.load(fpath)
        print(f"Loaded {name}: {fpath}")
    else:
        print(f"Not found: {fpath}")

if len(probs) < 2:
    print("Need at least 2 prob files for ensemble")
    sys.exit(1)

# Individual results
print("\n=== Individual Model Results ===")
for name, p in probs.items():
    bf, bt = 0, 0.5
    for t in np.arange(0.30, 0.70, 0.005):
        f1 = f1_score(
            y_dev, (p > t).astype(int),
            average='macro'
        )
        if f1 > bf:
            bf = f1
            bt = t
    print(f"  {name:15s}: F1={bf:.4f} (t={bt:.3f})")

# Try all combinations of 2+ models
from itertools import combinations

print("\n=== Ensemble Results ===")
names = list(probs.keys())
best_ens_f1 = 0
best_ens_preds = None
best_ens_name = ""

for r in range(2, len(names) + 1):
    for combo in combinations(names, r):
        avg_p = np.mean(
            [probs[n] for n in combo], axis=0
        )
        bf, bt = 0, 0.5
        for t in np.arange(0.30, 0.70, 0.005):
            f1 = f1_score(
                y_dev, (avg_p > t).astype(int),
                average='macro'
            )
            if f1 > bf:
                bf = f1
                bt = t
        ens_name = "+".join(combo)
        if bf > best_ens_f1:
            best_ens_f1 = bf
            best_ens_preds = (avg_p > bt).astype(int)
            best_ens_name = ens_name
        if bf >= 0.83:  # Only print promising ones
            print(
                f"  {ens_name:40s}: "
                f"F1={bf:.4f} (t={bt:.3f})"
            )

print(f"\n=== BEST ENSEMBLE ===")
print(f"  {best_ens_name}")
print(f"  F1={best_ens_f1:.4f}")

metrics = compute_all_metrics(y_dev, best_ens_preds)
print_metrics(metrics, "AV Cat C Ensemble — Final")

pred_path = (
    pred_dir / 'av_Group_34_C_ensemble.csv'
)
save_predictions(best_ens_preds, pred_path)
print(f"Saved to {pred_path}")

for n, bl in [('SVM', 0.5610), ('LSTM', 0.6226),
              ('BERT', 0.7854)]:
    gap = metrics['macro_f1'] - bl
    s = "BEATS" if gap > 0 else "BELOW"
    print(f"  vs {n}: {s} by {gap:+.4f}")
print(f"Current best AV Cat C: 0.8293")
print("Done!")
