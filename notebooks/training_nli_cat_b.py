"""
COMP34812 — Solution 2 (Category B) Training Notebook
Group 34 — NLI Track
"""

# %% [markdown]
# # Solution 2 (Category B) — Training
# ## ESIM + KIM for NLI
#
# Enhanced Sequential Inference Model with Knowledge-based Inference Model
# features. Uses BiLSTM encoding, soft cross-attention, WordNet knowledge
# injection, and character-level CNN for morphological robustness.

# %%
# !pip install torch numpy pandas

# %%
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.nli_cat_b_model import ESIM
from src.models.nli_cat_b_dataset import NLIVocabulary, NLIESIMDataset
from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics

# %% [markdown]
# ## Hyperparameters

# %%
HIDDEN_SIZE = 300
BATCH_SIZE = 32
MAX_EPOCHS = 40
PATIENCE = 7
LR = 4e-4
PREMISE_MAX = 64
HYPOTHESIS_MAX = 32
USE_WORDNET = True

# %% [markdown]
# ## 1. Load Data and Build Vocabulary

# %%
train_df = load_nli_data(split='train')
dev_df = load_nli_data(split='dev')
dev_labels = load_solution_labels(task='nli')
print(f"Train: {len(train_df)}, Dev: {len(dev_df)}")
