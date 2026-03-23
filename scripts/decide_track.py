#!/usr/bin/env python3
"""
Track Decision Framework — Apply decision rules
to choose between AV and NLI tracks.

Usage: python scripts/decide_track.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data_utils import load_solution_labels
from src.scorer import quick_score
from src.evaluation.eval_utils import mcnemars_test


def load_preds(path):
    preds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    preds.append(int(float(line)))
                except ValueError:
                    continue
    return np.array(preds)


# Baselines
AV_BASELINES = {'SVM': 0.5610, 'LSTM': 0.6226, 'BERT': 0.7854}
NLI_BASELINES = {'SVM': 0.5846, 'LSTM': 0.6603, 'BERT': 0.8198}


def evaluate_track(task, sol1_path, sol2_path):
    """Evaluate both solutions for a track."""
    y_true = np.array(load_solution_labels(task=task))
    baselines = AV_BASELINES if task == 'av' else NLI_BASELINES

    results = {}
    for name, path in [('Sol1', sol1_path), ('Sol2', sol2_path)]:
        if not Path(path).exists():
            print(f"  {name}: predictions not found at {path}")
            results[name] = None
            continue
        y_pred = load_preds(path)
        f1, mcc = quick_score(y_true, y_pred)
        cat = Path(path).stem.split('_')[-1]  # A, B, or C
        bl_name = 'SVM' if cat == 'A' else 'LSTM'
        bl_f1 = baselines[bl_name]
        gap = f1 - bl_f1
        results[name] = {'f1': f1, 'mcc': mcc, 'gap': gap, 'cat': cat}
        print(f"  {name} (Cat {cat}): F1={f1:.4f}, gap over {bl_name}={gap:+.4f}")
    return results
