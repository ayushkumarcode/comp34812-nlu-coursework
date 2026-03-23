"""Extract fitted NLI Cat A ensemble and evaluate.
Bypasses the stuck cross_val_score step.
"""
import sys
import signal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.nli_pipeline import NLIFeatureExtractor
from src.scorer import compute_all_metrics, print_metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier

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
