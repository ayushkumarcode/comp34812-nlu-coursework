"""NLI Cat A iter10 — XGB+LGBM with CalibratedCV + threshold tuning.

The v4 stack got 0.7103 with pruned features.
Now try: CalibratedClassifierCV on XGB and LGBM, then threshold search.
Also try: Gradient Boosting (sklearn) as third model.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.nli_pipeline import NLIFeatureExtractor
from src.scorer import compute_all_metrics, print_metrics

print("Loading...")
train_df = load_nli_data(split='train')
dev_df = load_nli_data(split='dev')
y_train = train_df['label'].values
y_dev = np.array(load_solution_labels(task='nli'))

print("Features...")
ext = NLIFeatureExtractor(use_spacy=True, use_glove=False)
ext.fit(train_df)
X_train, fnames = ext.transform(train_df)
X_dev, _ = ext.transform(dev_df)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_dv = scaler.transform(X_dev)

# Feature pruning (keep above-median)
print("Feature selection...")
sel = XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.05,
                     eval_metric='logloss', random_state=42, n_jobs=1)
sel.fit(X_tr, y_train)
imp = sel.feature_importances_
mask = imp >= np.median(imp)
X_tr_s, X_dv_s = X_tr[:, mask], X_dv[:, mask]
print(f"  Keeping {mask.sum()}/{len(mask)} features")

# Stacking with GBM + calibrated probabilities
print("Training XGB+LGBM+GBM stack (calibrated)...")
base = [
    ('xgb', CalibratedClassifierCV(
        XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.01,
                       subsample=0.7, colsample_bytree=0.6,
                       reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
