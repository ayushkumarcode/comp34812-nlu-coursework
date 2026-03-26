"""
AV Category A — Full training pipeline (LightGBM).
Run on CSF3: python -m src.training.run_av_cat_a

Trains LightGBM classifier with ~695 stylometric features.
Hyperparameters: n_estimators=1000, max_depth=7, lr=0.05,
num_leaves=63, subsample=0.8, colsample_bytree=0.8,
min_child_samples=20, reg_alpha=0.1, reg_lambda=1.
"""

import sys
import time
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.av_pipeline import AVFeatureExtractor
from src.scorer import compute_all_metrics, print_metrics


def main():
    print("=" * 60)
    print("  AV Category A — LightGBM Training")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    t0 = time.time()
    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    y_train = train_df['label'].values
    y_dev = np.array(load_solution_labels(task='av'))
    print(f"  Train: {len(train_df)}, Dev: {len(dev_df)}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Feature extraction
    print("\n[2/5] Extracting features...")
    t0 = time.time()
    extractor = AVFeatureExtractor(use_spacy=True, n_svd_components=100)
    extractor.fit(train_df)
    X_train, feature_names = extractor.transform(train_df)
    X_dev, _ = extractor.transform(dev_df)
    print(f"  Train: {X_train.shape}, Dev: {X_dev.shape}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Scale features
    print("\n[3/5] Training LightGBM...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_dv = scaler.transform(X_dev)

    model = LGBMClassifier(
        n_estimators=1000, max_depth=7, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=1,
        verbose=-1, random_state=42, n_jobs=1,
    )
    t0 = time.time()
    model.fit(X_tr, y_train)
    print(f"  Training time: {time.time() - t0:.1f}s")

    # Evaluate
    y_pred = model.predict(X_dv)
    metrics = compute_all_metrics(y_dev, y_pred)
    print_metrics(metrics, "AV Cat A — Dev Results")

    # Save model artifacts
    print("\n[4/5] Saving model...")
    save_dir = PROJECT_ROOT / 'models'
    save_dir.mkdir(exist_ok=True)
    joblib.dump(model, save_dir / 'av_cat_a_lgbm.joblib')
    joblib.dump(scaler, save_dir / 'av_cat_a_scaler.joblib')
    joblib.dump(feature_names, save_dir / 'av_cat_a_feature_names.joblib')
    joblib.dump(extractor.tfidf, save_dir / 'av_cat_a_tfidf.joblib')
    joblib.dump(extractor.cosine, save_dir / 'av_cat_a_cosine.joblib')
    print("  Saved all artifacts to models/")

    # Save predictions
    print("\n[5/5] Saving predictions...")
    pred_path = PROJECT_ROOT / 'predictions' / 'Group_34_A.csv'
    save_predictions(y_pred, pred_path)
    print(f"  Saved to {pred_path}")

    # Baseline comparison
    baselines = {'SVM': 0.5610, 'LSTM': 0.6226, 'BERT': 0.7854}
    f1 = metrics['macro_f1']
    for name, baseline_f1 in baselines.items():
        gap = f1 - baseline_f1
        print(f"  vs {name} ({baseline_f1:.4f}): {gap:+.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
