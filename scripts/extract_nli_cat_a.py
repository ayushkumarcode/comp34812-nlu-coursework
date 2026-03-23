"""Extract fitted NLI Cat A ensemble and evaluate.
Bypasses the stuck cross_val_score step.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.nli_pipeline import NLIFeatureExtractor
from src.scorer import compute_all_metrics, print_metrics
from sklearn.preprocessing import StandardScaler

print("Loading data...")
train_df = load_nli_data(split='train')
dev_df = load_nli_data(split='dev')
y_train = train_df['label'].values
y_dev = np.array(load_solution_labels(task='nli'))

print("Extracting features...")
extractor = NLIFeatureExtractor(use_spacy=True, use_glove=False)
extractor.fit(train_df)
X_train, feature_names = extractor.transform(train_df)
X_dev, _ = extractor.transform(dev_df)
print(f"Train: {X_train.shape}, Dev: {X_dev.shape}")

print("Scaling and training ensemble (no CV)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

from src.training.train_nli_ensemble import build_stacking_ensemble
ensemble = build_stacking_ensemble()
ensemble.fit(X_train_scaled, y_train)
print("Ensemble fitted.")

X_dev_scaled = scaler.transform(X_dev)
y_pred = ensemble.predict(X_dev_scaled)

metrics = compute_all_metrics(y_dev, y_pred)
print_metrics(metrics, "NLI Cat A — Dev Results")

pred_path = PROJECT_ROOT / 'predictions' / 'nli_Group_34_A.csv'
pred_path.parent.mkdir(exist_ok=True)
save_predictions(y_pred, pred_path)
print(f"Saved to {pred_path}")

import joblib
save_dir = PROJECT_ROOT / 'models'
save_dir.mkdir(exist_ok=True)
joblib.dump(scaler, save_dir / 'nli_cat_a_scaler.joblib')
joblib.dump(ensemble, save_dir / 'nli_cat_a_ensemble.joblib')
joblib.dump(extractor.tfidf, save_dir / 'nli_cat_a_tfidf.joblib')
joblib.dump(feature_names, save_dir / 'nli_cat_a_feature_names.joblib')
print("Models saved.")

baselines = {'SVM': 0.5846, 'LSTM': 0.6603, 'BERT': 0.8198}
for name, bl in baselines.items():
    gap = metrics['macro_f1'] - bl
    print(f"  vs {name}: {'BEATS' if gap > 0 else 'BELOW'} by {gap:+.4f}")
print("Done!")
