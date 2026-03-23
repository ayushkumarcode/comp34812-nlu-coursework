"""
COMP34812 — Solution 1 (Category A) Demo / Inference Notebook
Group 34 — NLI Track

Demonstrates how to load the trained model and make predictions.
"""

# %% [markdown]
# # Solution 1 (Category A) — Demo / Inference
# ## Feature-Rich Stacking Ensemble for Natural Language Inference

# %%
# !pip install scikit-learn xgboost lightgbm numpy pandas joblib spacy
# !python -m spacy download en_core_web_sm

# %%
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import clean_text, save_predictions
from src.nli_pipeline import NLIFeatureExtractor
from src.training.train_nli_ensemble import load_ensemble, predict, predict_proba

