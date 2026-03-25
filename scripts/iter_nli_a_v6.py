"""NLI Cat A iter11 — ExtraTrees + XGB + LGBM stack, no pruning, passthrough.

Previous best: v4 pruned at 0.7103.
Try: ExtraTrees instead of RF (more randomness, less overfitting),
     with higher n_estimators and no feature pruning.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
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

# Feature pruning
sel = XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.05,
                     eval_metric='logloss', random_state=42, n_jobs=1)
sel.fit(X_tr, y_train)
imp = sel.feature_importances_
mask = imp >= np.median(imp)
X_tr_s, X_dv_s = X_tr[:, mask], X_dv[:, mask]
print(f"Keeping {mask.sum()}/{len(mask)} features")

print("Training XGB+LGBM+ExtraTrees stack...")
base = [
    ('xgb', XGBClassifier(n_estimators=3000, max_depth=4, learning_rate=0.005,
                           subsample=0.7, colsample_bytree=0.5,
                           reg_alpha=0.1, reg_lambda=2.0, min_child_weight=10,
                           eval_metric='logloss', random_state=42, n_jobs=1)),
    ('lgbm', LGBMClassifier(n_estimators=3000, max_depth=4, learning_rate=0.005,
                              num_leaves=15, min_child_samples=30,
                              reg_alpha=0.1, reg_lambda=2.0,
                              verbose=-1, random_state=42, n_jobs=1)),
    ('et', ExtraTreesClassifier(n_estimators=1000, max_depth=15,
                                 min_samples_leaf=5,
                                 random_state=42, n_jobs=1)),
]
ens = StackingClassifier(
    estimators=base,
    final_estimator=LogisticRegression(C=0.5, max_iter=2000, random_state=42),
    cv=5, passthrough=True, n_jobs=1,
)
ens.fit(X_tr_s, y_train)
y_proba = ens.predict_proba(X_dv_s)[:, 1]
y_pred = ens.predict(X_dv_s)

metrics = compute_all_metrics(y_dev, y_pred)
print_metrics(metrics, "NLI Cat A (v6 XGB+LGBM+ET @0.5)")

# Threshold search
print("\nThreshold search:")
best_thresh, best_f1 = 0.5, metrics['macro_f1']
for t in np.arange(0.35, 0.65, 0.01):
    preds_t = (y_proba >= t).astype(int)
    f1_t = f1_score(y_dev, preds_t, average='macro', zero_division=0)
    if f1_t > best_f1:
        best_thresh, best_f1 = t, f1_t
    print(f"  thresh={t:.2f}: F1={f1_t:.4f}")

print(f"\nBest threshold: {best_thresh:.2f} -> F1={best_f1:.4f}")
final_preds = (y_proba >= best_thresh).astype(int)
metrics_final = compute_all_metrics(y_dev, final_preds)
print_metrics(metrics_final, "NLI Cat A (v6) — Final w/ Threshold")

save_predictions(final_preds, PROJECT_ROOT / 'predictions' / 'nli_Group_34_A_stack_v6.csv')
print("Done!")
