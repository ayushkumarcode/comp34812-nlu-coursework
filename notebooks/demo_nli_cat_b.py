"""
COMP34812 — Solution 2 (Category B) Demo / Inference Notebook
Group 34 — NLI Track

Demonstrates how to load the trained ESIM+KIM model and make predictions.
"""

# %% [markdown]
# # Solution 2 (Category B) — Demo / Inference
# ## ESIM + KIM: Enhanced Sequential Inference Model with WordNet Knowledge

# %%
# !pip install torch numpy pandas

# %%
import sys
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_nli_data, save_predictions
from src.models.nli_cat_b_model import ESIM
from src.models.nli_cat_b_dataset import NLIVocabulary, NLIESIMDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
