"""AV Cat A — XGB+LGBM stack."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.av_pipeline import AVFeatureExtractor
from src.scorer import compute_all_metrics, print_metrics

train_df = load_av_data(split='train')
dev_df = load_av_data(split='dev')
y_train = train_df['label'].values
y_dev = np.array(load_solution_labels(task='av'))
ext = AVFeatureExtractor(use_spacy=True, n_svd_components=100)
ext.fit(train_df)
X_train, _ = ext.transform(train_df)
X_dev, _ = ext.transform(dev_df)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_dv = scaler.transform(X_dev)

base = [
    ('xgb', XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           eval_metric='logloss', random_state=42, n_jobs=1)),
    ('lgbm', LGBMClassifier(n_estimators=500, max_depth=7, learning_rate=0.05,
                             num_leaves=63, verbose=-1, random_state=42, n_jobs=1)),
    ('lr', LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
]
