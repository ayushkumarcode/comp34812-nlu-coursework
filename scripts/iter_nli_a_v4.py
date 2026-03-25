"""NLI Cat A iteration 9 — XGB+LGBM+RF stack with feature importance pruning."""
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

# Step 1: Feature importance pruning with XGBoost
print("Step 1: Feature importance pruning...")
xgb_sel = XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.05,
                         subsample=0.7, colsample_bytree=0.7,
                         eval_metric='logloss', random_state=42, n_jobs=-1)
xgb_sel.fit(X_tr, y_train)
importances = xgb_sel.feature_importances_
# Keep features with above-median importance
median_imp = np.median(importances)
mask = importances >= median_imp
print(f"  Keeping {mask.sum()}/{len(mask)} features (median threshold={median_imp:.6f})")
X_tr_sel = X_tr[:, mask]
X_dv_sel = X_dv[:, mask]

# Step 2: Stack with selected features
print("Step 2: Training XGB+LGBM+RF stack (selected features)...")
base = [
    ('xgb', XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.01,
                           subsample=0.7, colsample_bytree=0.6, reg_alpha=0.1,
                           reg_lambda=1.0, min_child_weight=5,
                           eval_metric='logloss', random_state=42, n_jobs=1)),
    ('lgbm', LGBMClassifier(n_estimators=2000, max_depth=5, learning_rate=0.01,
                              num_leaves=31, min_child_samples=20,
                              reg_alpha=0.1, reg_lambda=1.0,
                              verbose=-1, random_state=42, n_jobs=1)),
    ('rf', RandomForestClassifier(n_estimators=500, max_depth=15,
                                   min_samples_leaf=5,
                                   random_state=42, n_jobs=1)),
]
ens = StackingClassifier(
    estimators=base,
    final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=42),
    cv=5, passthrough=True, n_jobs=1,
)
ens.fit(X_tr_sel, y_train)
y_pred = ens.predict(X_dv_sel)

metrics = compute_all_metrics(y_dev, y_pred)
print_metrics(metrics, "NLI Cat A (stack v4 — pruned XGB+LGBM+RF)")

save_predictions(y_pred, PROJECT_ROOT / 'predictions' / 'nli_Group_34_A_stack_v4.csv')
for n, bl in {'SVM': 0.5846, 'LSTM': 0.6603}.items():
    print(f"  vs {n}: {'+' if metrics['macro_f1']>bl else ''}{metrics['macro_f1']-bl:.4f}")

# Also try without pruning for comparison
print("\nStep 3: Training XGB+LGBM+RF stack (ALL features)...")
ens2 = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.01,
                               subsample=0.7, colsample_bytree=0.6, reg_alpha=0.1,
                               reg_lambda=1.0, min_child_weight=5,
                               eval_metric='logloss', random_state=42, n_jobs=1)),
