"""
COMP34812 — Solution 1 (Category A) Training Notebook
Group 34

Feature-rich stacking ensemble for Authorship Verification / NLI.
This notebook documents the complete training pipeline.

To convert: jupyter nbconvert --to notebook training_cat_a.py
"""

# %% [markdown]
# # Solution 1 (Category A) — Training
# ## Diff-Vector Stacking Ensemble with Comprehensive Stylometrics
#
# This notebook trains our Category A solution: a stacking ensemble
# (SVM-RBF, Random Forest, XGBoost) with logistic regression meta-learner,
# using ~950 stylometric features per text pair.

# %%
# !pip install scikit-learn xgboost lightgbm spacy numpy pandas tqdm joblib
# !python -m spacy download en_core_web_md

# %%
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.av_pipeline import AVFeatureExtractor
from src.training.train_av_ensemble import train_ensemble, save_ensemble, predict
from src.scorer import compute_all_metrics, print_metrics

# %% [markdown]
# ## 1. Load Data

# %%
train_df = load_av_data(split='train')
dev_df = load_av_data(split='dev')
y_train = train_df['label'].values
y_dev = np.array(load_solution_labels(task='av'))

print(f"Train: {len(train_df)} pairs")
print(f"Dev: {len(dev_df)} pairs")
print(f"Train label dist: {np.bincount(y_train)}")

# %% [markdown]
# ## 2. Feature Extraction
#
# We extract ~468 per-text features across 9 groups:
# - Lexical (30), Character (56), TF-IDF+SVD (100), Function words (150)
# - POS tags (45), Structural (15), Syntactic complexity (10, NOVEL)
# - Writing rhythm (6, NOVEL), Information-theoretic (5, NOVEL)
#
# Then compute diff-vectors |f(text1) - f(text2)| + style-only diff-vectors
# + 14 pairwise features = ~950 total features per pair.

# %%
extractor = AVFeatureExtractor(use_spacy=True, n_svd_components=100)
extractor.fit(train_df)

X_train, feature_names = extractor.transform(train_df)
X_dev, _ = extractor.transform(dev_df)

print(f"Train features: {X_train.shape}")
print(f"Dev features: {X_dev.shape}")
print(f"Feature count: {len(feature_names)}")

# %% [markdown]
# ## 3. Train Stacking Ensemble

# %%
scaler, ensemble, dev_metrics = train_ensemble(X_train, y_train, X_dev, y_dev)

# %% [markdown]
# ## 4. Save Model

# %%
save_ensemble(scaler, ensemble, extractor, save_dir='models')

# %% [markdown]
# ## 5. Generate Predictions

# %%
y_pred = predict(X_dev, scaler, ensemble)
save_predictions(y_pred, 'predictions/Group_34_A.csv')

metrics = compute_all_metrics(y_dev, y_pred)
print_metrics(metrics, "Category A — Dev Set Results")

# Baseline comparison
baselines = {'SVM': 0.5610, 'LSTM': 0.6226, 'BERT': 0.7854}
f1 = metrics['macro_f1']
for name, baseline in baselines.items():
    gap = f1 - baseline
    print(f"vs {name} ({baseline:.4f}): {'BEATS' if gap > 0 else 'BELOW'} by {gap:+.4f}")
