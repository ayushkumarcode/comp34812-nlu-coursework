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
