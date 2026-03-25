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
