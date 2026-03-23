"""AV Cat A — Simple XGBoost (no stacking, no SVM)."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
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
print_metrics(metrics, "AV Cat A (XGBoost)")

pred_path = PROJECT_ROOT / 'predictions' / 'av_Group_34_A.csv'
pred_path.parent.mkdir(exist_ok=True)
save_predictions(y_pred, pred_path)

save_dir = PROJECT_ROOT / 'models'
save_dir.mkdir(exist_ok=True)
joblib.dump(scaler, save_dir / 'av_cat_a_scaler.joblib')
joblib.dump(model, save_dir / 'av_cat_a_ensemble.joblib')

baselines = {'SVM': 0.5610, 'LSTM': 0.6226, 'BERT': 0.7854}
for n, bl in baselines.items():
    gap = metrics['macro_f1'] - bl
    print(f"  vs {n}: {'BEATS' if gap > 0 else 'BELOW'} by {gap:+.4f}")
print("Done!")
