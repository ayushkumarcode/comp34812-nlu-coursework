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
