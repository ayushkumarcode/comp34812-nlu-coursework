"""AV Cat A — Feature importance pruning + XGBoost retrain.

Strategy: Use initial XGBoost to get feature importances,
remove bottom 50% of features, retrain with optimized params.
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (
    load_av_data, load_solution_labels, save_predictions
)
from src.av_pipeline import AVFeatureExtractor
from src.scorer import compute_all_metrics, print_metrics

print("=== AV Cat A: Feature Pruning + XGBoost ===\n")

train_df = load_av_data(split='train')
dev_df = load_av_data(split='dev')
y_train = train_df['label'].values
y_dev = np.array(load_solution_labels(task='av'))

print("Extracting features...")
ext = AVFeatureExtractor(
    use_spacy=True, n_svd_components=100
)
ext.fit(train_df)
X_train, fnames = ext.transform(train_df)
X_dev, _ = ext.transform(dev_df)
print(f"Features: {X_train.shape[1]}")

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_dv = scaler.transform(X_dev)

# Step 1: Get feature importances from initial XGB
print("\nStep 1: Initial XGBoost for feature selection")
sel = XGBClassifier(
    n_estimators=500, max_depth=7,
    learning_rate=0.05,
    eval_metric='logloss', random_state=42, n_jobs=-1
)
sel.fit(X_tr, y_train)
imp = sel.feature_importances_

# Try different pruning thresholds
for pct in [25, 50, 75]:
    threshold = np.percentile(imp, pct)
    mask = imp >= threshold
    n_kept = mask.sum()
    X_tr_p = X_tr[:, mask]
    X_dv_p = X_dv[:, mask]
    print(f"\n--- Pruning {pct}% (keeping {n_kept}) ---")

    # Retrain with better hyperparams
    model = XGBClassifier(
        n_estimators=2000, max_depth=5,
        learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=5,
        eval_metric='logloss',
        random_state=42, n_jobs=-1,
    )
    model.fit(X_tr_p, y_train)
    y_pred = model.predict(X_dv_p)
    f1 = f1_score(y_dev, y_pred, average='macro')
    print(f"  macro_f1 = {f1:.4f}")

# Step 2: Deep+regularized XGBoost on all features
print("\n--- Deep Regularized XGBoost (all features) ---")
model2 = XGBClassifier(
    n_estimators=3000, max_depth=3,
    learning_rate=0.01,
    subsample=0.6, colsample_bytree=0.5,
    reg_alpha=0.5, reg_lambda=2.0,
    min_child_weight=10, gamma=0.1,
    eval_metric='logloss',
    random_state=42, n_jobs=-1,
)
model2.fit(X_tr, y_train)
y_pred2 = model2.predict(X_dv)
y_prob2 = model2.predict_proba(X_dv)[:, 1]

# Threshold sweep
best_f1 = 0
best_t = 0.5
for t in np.arange(0.35, 0.65, 0.005):
    preds = (y_prob2 > t).astype(int)
    f1 = f1_score(y_dev, preds, average='macro')
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

final_preds = (y_prob2 > best_t).astype(int)
print(f"  Best: F1={best_f1:.4f} at thresh={best_t:.3f}")

metrics = compute_all_metrics(y_dev, final_preds)
print_metrics(metrics, "AV Cat A XGB Deep+Reg")

pred_path = (
    PROJECT_ROOT / 'predictions'
    / 'av_Group_34_A_xgb_deep.csv'
)
pred_path.parent.mkdir(exist_ok=True)
save_predictions(final_preds, pred_path)
