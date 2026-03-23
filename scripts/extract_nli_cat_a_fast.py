"""NLI Cat A — Fast extraction without SVM.
Drops SVM-RBF to avoid the multi-hour bottleneck.
Uses XGBoost + LightGBM + LR stacking instead.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.nli_pipeline import NLIFeatureExtractor
from src.scorer import compute_all_metrics, print_metrics

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

print("Training fast ensemble (no SVM)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

base = [
    ('xgb', XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           eval_metric='logloss', random_state=42, n_jobs=-1)),
    ('lgbm', LGBMClassifier(n_estimators=500, max_depth=7, learning_rate=0.05,
                             num_leaves=63, verbose=-1, random_state=42, n_jobs=-1)),
    ('lr', LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
]
ensemble = StackingClassifier(
    estimators=base,
    final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    cv=5, passthrough=False, n_jobs=-1,
)
ensemble.fit(X_train_scaled, y_train)
print("Ensemble fitted.")
