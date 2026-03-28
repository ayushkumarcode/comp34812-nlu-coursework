"""
Wrapper around the local scorer so we can compute all 8 official metrics
without having to invoke the CLI every time.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn import metrics as sklearn_metrics

from src.data_utils import SCORER_ROOT, save_predictions


def compute_all_metrics(y_true, y_pred):
    """Compute all 8 metrics the official scorer uses.

    Args:
        y_true: ground truth (0/1).
        y_pred: predictions (0/1).

    Returns:
        dict of metric name -> value.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    return {
        'accuracy_score': sklearn_metrics.accuracy_score(y_true, y_pred),
        'macro_precision': sklearn_metrics.precision_score(
            y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': sklearn_metrics.recall_score(
            y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': sklearn_metrics.f1_score(
            y_true, y_pred, average='macro', zero_division=0),
        'weighted_macro_precision': sklearn_metrics.precision_score(
            y_true, y_pred, average='weighted', zero_division=0),
        'weighted_macro_recall': sklearn_metrics.recall_score(
            y_true, y_pred, average='weighted', zero_division=0),
        'weighted_mmacro_f1': sklearn_metrics.f1_score(
            y_true, y_pred, average='weighted', zero_division=0),
        'matthews_corrcoef': sklearn_metrics.matthews_corrcoef(y_true, y_pred),
    }


def print_metrics(metrics, title=None):
    """Pretty-print a metrics dict.

    Args:
        metrics: dict from compute_all_metrics.
        title: optional header to print.
    """
    if title:
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}")
    for name, value in metrics.items():
        print(f"  {name:30s}: {value:.6f}")


def score_predictions_file(pred_path, task='av'):
    """Score a prediction file against the official reference data.
    Uses the same refs as the CLI scorer.

    Args:
        pred_path: path to a file with one 0/1 per line.
        task: 'av' or 'nli'.
    """
    scorer_path = str(SCORER_ROOT)
    if scorer_path not in sys.path:
        sys.path.insert(0, scorer_path)

    from local_scorer.io_utils import resolve_reference_path, read_numeric_array
    from local_scorer.metrics import compute_metrics, load_metric_names

    ref_path = resolve_reference_path(task)
    solution = read_numeric_array(ref_path)
    prediction = read_numeric_array(str(pred_path))

    metric_names = load_metric_names(
        SCORER_ROOT / "local_scorer" / "metric.txt"
    )
    scores = compute_metrics(solution, prediction, metric_names)
    return {name: value for name, value in scores}


def quick_score(y_true, y_pred):
    """Return the two most important metrics: macro_f1 and MCC.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Tuple of (macro_f1, mcc).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    f1 = sklearn_metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = sklearn_metrics.matthews_corrcoef(y_true, y_pred)
    return f1, mcc
