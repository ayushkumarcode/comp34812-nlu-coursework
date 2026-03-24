"""NLI Cat A — XGBoost with GloVe embedding features.

Same as nli_cat_a_simple.py but with use_glove=True to add GloVe-based
semantic similarity features (WMD approximation, IDF-weighted cosine,
embedding centroid similarity, etc.).
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.nli_pipeline import NLIFeatureExtractor
from src.scorer import compute_all_metrics, print_metrics

print("Loading data...")
train_df = load_nli_data(split='train')
dev_df = load_nli_data(split='dev')
y_train = train_df['label'].values
y_dev = np.array(load_solution_labels(task='nli'))

print("Extracting features (with GloVe)...")
ext = NLIFeatureExtractor(use_spacy=True, use_glove=True)
ext.fit(train_df)
X_train, fnames = ext.transform(train_df)
X_dev, _ = ext.transform(dev_df)
print(f"Train: {X_train.shape}, Dev: {X_dev.shape}")
print(f"Features: {len(fnames)}")

print("Training XGBoost...")
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_dv = scaler.transform(X_dev)

model = XGBClassifier(
    n_estimators=1000, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=5, gamma=0.1,
    eval_metric='logloss', random_state=42, n_jobs=1,
)
model.fit(X_tr, y_train)
y_pred = model.predict(X_dv)

metrics = compute_all_metrics(y_dev, y_pred)
print_metrics(metrics, "NLI Cat A (XGBoost + GloVe) — Dev")

pred_path = PROJECT_ROOT / 'predictions' / 'nli_Group_34_A_glove.csv'
pred_path.parent.mkdir(exist_ok=True)
save_predictions(y_pred, pred_path)

save_dir = PROJECT_ROOT / 'models'
save_dir.mkdir(exist_ok=True)
