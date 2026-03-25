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
