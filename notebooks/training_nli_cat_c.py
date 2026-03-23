"""
COMP34812 — Solution 2 (Category C) Training
Group 34 — NLI Track
"""

# %% [markdown]
# # Solution 2 (Category C) — Training
# ## DeBERTa-v3-base Cross-Encoder for NLI
#
# Fine-tuned DeBERTa-v3-base as a cross-encoder with hypothesis-only
# adversarial debiasing via Gradient Reversal Layer (GRL).
# Achieves F1=0.9167 on dev set, beating BERT baseline by +0.097.

# %%
# !pip install torch transformers numpy pandas

# %%
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_nli_data, load_solution_labels
from src.models.cat_c_deberta import NLIDeBERTaCrossEncoder
from src.scorer import compute_all_metrics, print_metrics
