"""Get probability outputs for NLI Cat A (XGBoost ensemble) for threshold optimization.

Uses predict_proba from sklearn-compatible models to get class probabilities,
then sweeps thresholds to find optimal decision boundary.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import joblib
from sklearn.metrics import f1_score

from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.nli_pipeline import NLIFeatureExtractor


def main():
    print("=== NLI Cat A Threshold Optimization ===\n")

    # Load model and scaler
    scaler = joblib.load(PROJECT_ROOT / 'models' / 'nli_cat_a_scaler.joblib')
    model = joblib.load(PROJECT_ROOT / 'models' / 'nli_cat_a_ensemble.joblib')
    print(f"Model type: {type(model).__name__}")

    # Load data
