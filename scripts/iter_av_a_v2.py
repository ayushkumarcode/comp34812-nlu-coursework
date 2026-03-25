"""AV Cat A iter4 — XGB+LGBM stack with more trees and lower LR.

Changes from v1:
- XGB n_estimators=2000, learning_rate=0.01, max_depth=5
- LGBM n_estimators=2000, learning_rate=0.01, max_depth=5
- Add RF as third base learner
- LogReg meta-learner with passthrough
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.av_pipeline import AVFeatureExtractor
from src.scorer import compute_all_metrics, print_metrics

print("Loading data...")
train_df = load_av_data(split='train')
dev_df = load_av_data(split='dev')
y_train = train_df['label'].values
y_dev = np.array(load_solution_labels(task='av'))

print("Extracting features...")
ext = AVFeatureExtractor(use_spacy=True, n_svd_components=100)
ext.fit(train_df)
X_train, fnames = ext.transform(train_df)
X_dev, _ = ext.transform(dev_df)
print(f"Train: {X_train.shape}, Dev: {X_dev.shape}")

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_dv = scaler.transform(X_dev)

# Feature importance pruning
print("Feature selection via XGBoost importance...")
sel = XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.05,
                     eval_metric='logloss', random_state=42, n_jobs=1)
sel.fit(X_tr, y_train)
imp = sel.feature_importances_
mask = imp >= np.median(imp)
print(f"  Keeping {mask.sum()}/{len(mask)} features")
X_tr_s, X_dv_s = X_tr[:, mask], X_dv[:, mask]

print("Training XGB+LGBM+RF stack (pruned features)...")
base = [
    ('xgb', XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.01,
                           subsample=0.7, colsample_bytree=0.6,
                           reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
                           eval_metric='logloss', random_state=42, n_jobs=1)),
    ('lgbm', LGBMClassifier(n_estimators=2000, max_depth=5, learning_rate=0.01,
                              num_leaves=31, min_child_samples=20,
                              reg_alpha=0.1, reg_lambda=1.0,
                              verbose=-1, random_state=42, n_jobs=1)),
    ('rf', RandomForestClassifier(n_estimators=500, max_depth=15,
                                   min_samples_leaf=5, random_state=42, n_jobs=1)),
]
ens = StackingClassifier(
    estimators=base,
    final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=42),
    cv=5, passthrough=True, n_jobs=1,
)
ens.fit(X_tr_s, y_train)
y_pred = ens.predict(X_dv_s)

metrics = compute_all_metrics(y_dev, y_pred)
print_metrics(metrics, "AV Cat A (stack v2 — pruned)")
save_predictions(y_pred, PROJECT_ROOT / 'predictions' / 'av_Group_34_A_stack_v2.csv')

# Also try full features
print("\nTraining XGB+LGBM+RF stack (ALL features)...")
ens2 = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.01,
                               subsample=0.7, colsample_bytree=0.6,
                               eval_metric='logloss', random_state=42, n_jobs=1)),
        ('lgbm', LGBMClassifier(n_estimators=2000, max_depth=5, learning_rate=0.01,
                                  num_leaves=31, verbose=-1, random_state=42, n_jobs=1)),
