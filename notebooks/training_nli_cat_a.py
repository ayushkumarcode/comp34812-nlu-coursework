"""
COMP34812 — Solution 1 (Category A) Training Notebook
Group 34 — NLI Track
"""

# %% [markdown]
# # Solution 1 (Category A) — Training
# ## Feature-Rich Stacking Ensemble for NLI
#
# This notebook trains a stacking ensemble (XGBoost, LightGBM, SVM-RBF, LR)
# with ~280 features per premise-hypothesis pair including alignment,
# natural logic, and interaction features.

# %%
# !pip install scikit-learn xgboost lightgbm spacy numpy pandas tqdm joblib
# !python -m spacy download en_core_web_sm

# %%
import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.nli_pipeline import NLIFeatureExtractor
from src.training.train_nli_ensemble import train_ensemble, save_ensemble, predict
from src.scorer import compute_all_metrics, print_metrics

# %% [markdown]
# ## 1. Load Data

# %%
train_df = load_nli_data(split='train')
dev_df = load_nli_data(split='dev')
y_train = train_df['label'].values
y_dev = np.array(load_solution_labels(task='nli'))

print(f"Train: {len(train_df)} pairs")
print(f"Dev: {len(dev_df)} pairs")
print(f"Train label dist: {np.bincount(y_train)}")

# %% [markdown]
# ## 2. Feature Extraction
#
# ~280 features: lexical overlap, semantic similarity,
# negation/contradiction, syntactic, alignment, natural logic,
# cross-sentence, TF-IDF+SVD, interaction terms.

# %%
extractor = NLIFeatureExtractor(use_spacy=True)
extractor.fit(train_df)

X_train, feature_names = extractor.transform(train_df)
X_dev, _ = extractor.transform(dev_df)
print(f"Train: {X_train.shape}, Dev: {X_dev.shape}")

# %% [markdown]
# ## 3. Train Stacking Ensemble

# %%
scaler, ensemble, dev_metrics = train_ensemble(X_train, y_train, X_dev, y_dev)

# %% [markdown]
# ## 4. Save Model and Generate Predictions

# %%
save_ensemble(scaler, ensemble, extractor, save_dir='models')
y_pred = predict(X_dev, scaler, ensemble)
save_predictions(y_pred, 'predictions/Group_34_A.csv')

metrics = compute_all_metrics(y_dev, y_pred)
print_metrics(metrics, "NLI Cat A — Dev Set Results")

baselines = {'SVM': 0.5846, 'LSTM': 0.6603, 'BERT': 0.8198}
for name, baseline in baselines.items():
    gap = metrics['macro_f1'] - baseline
    print(f"vs {name} ({baseline:.4f}): {'BEATS' if gap > 0 else 'BELOW'} by {gap:+.4f}")
