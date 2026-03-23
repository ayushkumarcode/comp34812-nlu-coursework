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

# %% [markdown]
# ## Architecture
#
# - **Encoder**: DeBERTa-v3-base (microsoft/deberta-v3-base)
# - **Input**: [CLS] premise [SEP] hypothesis [SEP]
# - **Classifier**: Dropout(0.1) -> Linear(768, 256) -> Tanh -> Dropout(0.1) -> Linear(256, 1)
# - **Adversarial Debiasing**: Hypothesis-only encoder + GRL (lambda=0.1) to prevent
#   hypothesis-only shortcuts (McCoy et al. 2019)
# - **Training**: AdamW (lr=2e-5), mixed precision, early stopping (patience=5)
# - **Loss**: BCE + 0.1 * adversarial BCE

# %% [markdown]
# ## Training
# Training is done on CSF3 GPU via:
# ```bash
# sbatch scripts/train_nli_cat_c.sh
# ```
# See `src/training/train_cat_c.py` for the full training code.

# %% [markdown]
# ## Results
# Best dev macro_f1 = 0.9167, MCC = 0.8339
# Beats all baselines with statistical significance (p < 0.001).
