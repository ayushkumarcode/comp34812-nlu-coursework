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
