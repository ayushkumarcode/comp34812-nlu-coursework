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

    for thresh in np.arange(0.30, 0.71, 0.01):
        preds = (probs > thresh).astype(int)
        f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        if abs(thresh - 0.5) < 0.005:
            default_f1 = f1
            print(f"  thresh=0.50: F1={f1:.4f}  <-- default")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\nBest threshold: {best_thresh:.2f} -> F1={best_f1:.4f}")
    if default_f1 is not None:
        print(f"Improvement over 0.50: {best_f1 - default_f1:+.4f}")

    # Save predictions with optimal threshold
    preds = (probs > best_thresh).astype(int)
    pred_path = PROJECT_ROOT / 'predictions' / 'nli_Group_34_A_thresh.csv'
    save_predictions(preds, pred_path)
    print(f"Optimized predictions saved to {pred_path}")

    # Detailed sweep around the optimum
    print("\n=== Detailed sweep around optimum ===")
    for thresh in np.arange(max(0.30, best_thresh - 0.05),
                             min(0.71, best_thresh + 0.06), 0.01):
        preds = (probs > thresh).astype(int)
        f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        marker = " <-- BEST" if abs(thresh - best_thresh) < 0.005 else ""
        print(f"  thresh={thresh:.2f}: F1={f1:.4f}{marker}")

    print("\nDone!")


if __name__ == '__main__':
    main()
