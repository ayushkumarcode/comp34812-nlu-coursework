"""Score all AV prediction files and print summary.

Loads all prediction CSV files and probability NPY files,
computes F1 scores, and provides a comparison table.
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_solution_labels

y_dev = np.array(load_solution_labels(task='av'))
pred_dir = PROJECT_ROOT / 'predictions'

print("=" * 65)
print("  AV Prediction Scores Summary")
print("=" * 65)

# Score CSV prediction files
print("\n--- CSV Predictions (hard labels) ---")
results = []
for f in sorted(pred_dir.glob('av_*.csv')):
    try:
        preds = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    preds.append(int(float(line)))
        if len(preds) != len(y_dev):
            continue
        preds = np.array(preds)
        f1 = f1_score(y_dev, preds, average='macro')
        results.append((f.name, f1))
        cat = 'A' if '_A' in f.name else (
            'B' if '_B' in f.name else 'C'
        )
        print(f"  Cat {cat} | {f.name:45s} | F1={f1:.4f}")
    except Exception as e:
        print(f"  ERROR: {f.name}: {e}")

# Score NPY probability files
print("\n--- NPY Probabilities (with threshold) ---")
for f in sorted(pred_dir.glob('av_*probs*.npy')):
    try:
        p = np.load(f)
        if len(p) != len(y_dev):
            continue
        bf, bt = 0, 0.5
        for t in np.arange(0.30, 0.70, 0.005):
            f1 = f1_score(
                y_dev, (p > t).astype(int),
                average='macro'
            )
            if f1 > bf:
                bf = f1
                bt = t
        print(
            f"  {f.name:45s} | "
            f"F1={bf:.4f} (t={bt:.3f})"
        )
    except Exception as e:
        print(f"  ERROR: {f.name}: {e}")

# Best per category
print("\n--- Best Per Category ---")
cat_best = {'A': 0, 'B': 0, 'C': 0}
for name, f1 in results:
    cat = 'A' if '_A' in name else (
        'B' if '_B' in name else 'C'
    )
    if f1 > cat_best[cat]:
        cat_best[cat] = f1

baselines = {
    'SVM': 0.5610, 'LSTM': 0.6226, 'BERT': 0.7854
}
for cat in ['A', 'B', 'C']:
    bf = cat_best[cat]
    print(f"\n  Category {cat}: best F1 = {bf:.4f}")
    for bn, bv in baselines.items():
        gap = bf - bv
        s = "BEATS" if gap > 0 else "BELOW"
        print(f"    vs {bn} ({bv:.4f}): {s} {gap:+.4f}")

print("\nDone!")
