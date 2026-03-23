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

# %% [markdown]
# ## 3. Generate Predictions

# %%
all_preds, all_probs = [], []
with torch.no_grad():
    for batch in test_loader:
        p_word = batch['premise_word_ids'].to(device)
        p_char = batch['premise_char_ids'].to(device)
        h_word = batch['hypothesis_word_ids'].to(device)
        h_char = batch['hypothesis_char_ids'].to(device)
        logits = model(p_word, p_char, h_word, h_char)
        probs = torch.sigmoid(logits.squeeze(-1))
        preds = (probs > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

predictions = np.array(all_preds)
probabilities = np.array(all_probs)
print(f"Predictions: {len(predictions)}")
print(f"Class distribution: {np.bincount(predictions)}")

# %% [markdown]
# ## 4. Save Predictions

# %%
save_predictions(predictions, 'predictions/Group_34_B.csv')
print("Predictions saved to predictions/Group_34_B.csv")

# %% [markdown]
# ## 5. Example Predictions

# %%
for i in range(min(5, len(test_df))):
    premise = test_df.iloc[i]['premise'][:100] + "..."
    hypothesis = test_df.iloc[i]['hypothesis'][:80]
    label = "Entailed" if predictions[i] == 1 else "Not Entailed"
    print(f"\nPair {i+1}:")
    print(f"  Premise:    {premise}")
    print(f"  Hypothesis: {hypothesis}")
    print(f"  Prediction: {label} (prob={probabilities[i]:.3f})")
