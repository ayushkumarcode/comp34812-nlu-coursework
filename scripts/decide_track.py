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
