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
    task_name = 'Authorship Verification' if task == 'av' else 'NLI'
    ax.set_title(f'{task_name} — F1 Comparison', fontsize=16)
    ax.set_ylim(0, max(scores) * 1.15)
    plt.tight_layout()
    if save:
        path = POSTER_DIR / 'f1_chart.png'
        plt.savefig(str(path), dpi=200, bbox_inches='tight')
        print(f"Saved to {path}")
    return fig


def confusion_matrix_heatmap(y_true, y_pred, title='Confusion Matrix', save_name='cm.png'):
    """Generate confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Class 0', 'Class 1'], fontsize=12)
    ax.set_yticklabels(['Class 0', 'Class 1'], fontsize=12)
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('True', fontsize=14)
    ax.set_title(title, fontsize=16)
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=18, fontweight='bold', color=color)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = POSTER_DIR / save_name
    plt.savefig(str(path), dpi=200, bbox_inches='tight')
    print(f"Saved to {path}")
    return fig


if __name__ == '__main__':
    # Test with AV results
    f1_bar_chart(
        results={'Sol 1\n(Cat A)': 0.72, 'Sol 2\n(Cat B)': 0.7123},
        baselines={'SVM': 0.5610, 'LSTM': 0.6226, 'BERT': 0.7854},
        task='av',
    )
