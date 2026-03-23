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
print(f"Device: {device}")

# %% [markdown]
# ## 1. Load Trained Model

# %%
checkpoint = torch.load('models/nli_cat_b_best.pt',
                         map_location=device, weights_only=False)
vocab = checkpoint['vocab']
config = checkpoint['config']

model = ESIM(
    vocab_size=vocab.vocab_size,
    embedding_dim=config['emb_dim'],
    hidden_size=config['hidden_size'],
    char_vocab_size=vocab.char_vocab_size,
    knowledge_dim=5, dropout=0.0,
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded. Vocab: {vocab.vocab_size} words")

# %% [markdown]
# ## 2. Prepare Test Data

# %%
from torch.utils.data import DataLoader

test_df = load_nli_data(split='dev')
test_dataset = NLIESIMDataset(
    test_df, vocab,
    premise_max_len=config['premise_max'],
    hypothesis_max_len=config['hypothesis_max'],
    compute_wordnet=False,
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Test data: {len(test_df)} pairs")
