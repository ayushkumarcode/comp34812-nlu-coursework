#!/usr/bin/env python3
"""Generate matplotlib visualizations for the poster."""

import sys
from pathlib import Path

VENV = Path('/tmp/poster_env')
if VENV.exists():
    for p in VENV.glob('lib/*/site-packages'):
        sys.path.insert(0, str(p))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

POSTER_DIR = Path(__file__).parent


def f1_bar_chart(results, baselines, task='av', save=True):
    """Generate F1 comparison bar chart.

    Args:
        results: dict like {'Sol 1 (Cat A)': 0.72, 'Sol 2 (Cat B)': 0.71}
        baselines: dict like {'SVM': 0.56, 'LSTM': 0.62, 'BERT': 0.79}
        task: 'av' or 'nli'
        save: save to poster/f1_chart.png
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(baselines.keys()) + list(results.keys())
    scores = list(baselines.values()) + list(results.values())
    n_bl = len(baselines)
    colors = ['#95a5a6'] * n_bl + ['#2ecc71', '#e74c3c']
    bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro F1 Score', fontsize=14)
