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
    train_df = load_nli_data(split='train')
    dev_df = load_nli_data(split='dev')
    y_true = np.array(load_solution_labels(task='nli'))

    # Extract features
    ext = NLIFeatureExtractor(use_spacy=True, use_glove=False)
    ext.fit(train_df)
    X_dev, feature_names = ext.transform(dev_df)
    X_dv = scaler.transform(X_dev)
    print(f"Feature matrix: {X_dv.shape}")

    # Get probabilities
    probs = model.predict_proba(X_dv)[:, 1]
    print(f"Prob stats: min={probs.min():.4f}, max={probs.max():.4f}, "
          f"mean={probs.mean():.4f}, std={probs.std():.4f}")

    # Save probabilities
    prob_path = PROJECT_ROOT / 'predictions' / 'nli_cat_a_probs.npy'
    prob_path.parent.mkdir(exist_ok=True)
    np.save(str(prob_path), probs)
    print(f"Probabilities saved to {prob_path}")

    # Sweep thresholds
    print("\n=== Threshold Sweep ===")
    best_f1 = 0
    best_thresh = 0.5
    default_f1 = None

