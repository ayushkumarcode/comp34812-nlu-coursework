"""AV Cat A — Comprehensive model search using cached features.

Loads pre-cached numpy feature arrays (much faster than
re-extracting). Tries many classifier configs.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import save_predictions
from src.scorer import compute_all_metrics, print_metrics

print("=== AV Cat A: Cached Feature Search ===\n")

cache = PROJECT_ROOT / 'models'
try:
    X_tr = np.load(cache / 'av_features_train.npy')
    X_dv = np.load(cache / 'av_features_dev.npy')
    y_tr = np.load(cache / 'av_labels_train.npy')
    y_dv = np.load(cache / 'av_labels_dev.npy')
    print(f"Loaded cached features: {X_tr.shape}")
except FileNotFoundError:
    print("Cached features not found!")
    print("Run cache_av_features.py first")
    sys.exit(1)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_dv = scaler.transform(X_dv)

results = {}

# 1. XGB configurations
configs_xgb = [
    {'n': 1500, 'd': 5, 'lr': 0.02, 'sub': 0.7,
     'col': 0.6, 'ra': 0.1, 'rl': 1.0, 'mcw': 5},
    {'n': 3000, 'd': 3, 'lr': 0.01, 'sub': 0.6,
