"""
COMP34812 — Solution 2 (Cat C) Demo
Group 34 — NLI Track

DeBERTa-v3-base cross-encoder for NLI.
"""

# %% [markdown]
# # Solution 2 (Category C) — Demo
# ## DeBERTa-v3-base Cross-Encoder for NLI

# %%
# !pip install torch transformers numpy pandas

# %%
import sys
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_nli_data, save_predictions
from src.models.cat_c_deberta import NLIDeBERTaCrossEncoder
from transformers import AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# %% [markdown]
# ## 1. Load Trained Model

# %%
MODEL_NAME = 'microsoft/deberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

model = NLIDeBERTaCrossEncoder(model_name=MODEL_NAME).to(device)
model.load_state_dict(torch.load('models/nli_cat_c_best.pt',
                                  map_location=device, weights_only=True))
