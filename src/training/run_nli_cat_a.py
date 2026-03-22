"""
NLI Category A — Full training pipeline.
Run on CSF3: python -m src.training.run_nli_cat_a
"""

import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.nli_pipeline import NLIFeatureExtractor
from src.training.train_nli_ensemble import (
    train_ensemble, save_ensemble, predict
)
from src.scorer import compute_all_metrics, print_metrics


def main():
    print("=" * 60)
    print("  NLI Category A — Full Training Pipeline")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    t0 = time.time()
    train_df = load_nli_data(split='train')
    dev_df = load_nli_data(split='dev')
    print(f"  Train: {len(train_df)} pairs")
    print(f"  Dev: {len(dev_df)} pairs")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Load dev labels
    dev_labels = load_solution_labels(task='nli')
    y_train = train_df['label'].values
    y_dev = np.array(dev_labels)
    print(f"  Dev labels: {len(y_dev)} samples")

    # Feature extraction
    print("\n[2/5] Extracting features...")
    t0 = time.time()
    extractor = NLIFeatureExtractor(use_spacy=True, use_glove=False)
    extractor.fit(train_df)
    X_train, feature_names = extractor.transform(train_df)
    X_dev, _ = extractor.transform(dev_df)
    print(f"  Train features: {X_train.shape}")
    print(f"  Dev features: {X_dev.shape}")
    print(f"  Feature count: {len(feature_names)}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Train ensemble
    print("\n[3/5] Training stacking ensemble...")
    t0 = time.time()
    scaler, ensemble, dev_metrics = train_ensemble(
        X_train, y_train, X_dev, y_dev
    )
    print(f"  Time: {time.time() - t0:.1f}s")

    # Save model
    print("\n[4/5] Saving model...")
    save_dir = PROJECT_ROOT / 'models'
    save_ensemble(scaler, ensemble, extractor, save_dir=str(save_dir))

    # Generate predictions file
    print("\n[5/5] Generating prediction file...")
    y_pred = predict(X_dev, scaler, ensemble)
    pred_path = PROJECT_ROOT / 'predictions' / 'nli_Group_34_A.csv'
    pred_path.parent.mkdir(exist_ok=True)
    save_predictions(y_pred, pred_path)
    print(f"  Saved predictions to {pred_path}")

    # Final metrics
    metrics = compute_all_metrics(y_dev, y_pred)
    print_metrics(metrics, "NLI Cat A — Final Dev Results")

    # Baseline comparison
    print("\n" + "=" * 60)
    print("  Baseline Comparison")
    print("=" * 60)
    baselines = {'SVM': 0.5846, 'LSTM': 0.6603, 'BERT': 0.8198}
    f1 = metrics['macro_f1']
    for name, baseline_f1 in baselines.items():
        gap = f1 - baseline_f1
        status = "BEATS" if gap > 0 else "BELOW"
        print(f"  vs {name} ({baseline_f1:.4f}): {status} by {gap:+.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
