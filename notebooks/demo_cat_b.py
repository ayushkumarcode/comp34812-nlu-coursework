"""
COMP34812 — Solution 2 (Category B) Demo / Inference Notebook
Group 34

Demonstrates how to load the trained neural model and make predictions.
"""

# %% [markdown]
# # Solution 2 (Category B) — Demo / Inference
# ## Loading the trained model and making predictions

# %%
# !pip install torch numpy pandas

# %%
import sys
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_av_data, save_predictions
from src.models.av_cat_b_model import AVCatBModel
from src.models.av_cat_b_dataset import AVCharDataset, VOCAB_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# %% [markdown]
# ## 1. Load Trained Model

# %%
model = AVCatBModel(
    vocab_size=VOCAB_SIZE, char_emb_dim=32,
    cnn_filters=128, lstm_hidden=128,
    proj_dim=128, num_topics=10,
).to(device)

model.load_state_dict(torch.load('models/av_cat_b_best.pt',
                                  map_location=device, weights_only=True))
model.eval()
print("Model loaded successfully.")

# %% [markdown]
# ## 2. Prepare Test Data

# %%
from torch.utils.data import DataLoader

test_df = load_av_data(split='dev')
test_dataset = AVCharDataset(test_df, max_len=1500, augment=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# %% [markdown]
# ## 3. Generate Predictions

# %%
all_preds = []
all_probs = []

with torch.no_grad():
    for batch in test_loader:
        char_1 = batch['char_ids_1'].to(device)
        char_2 = batch['char_ids_2'].to(device)

        logits, _ = model(char_1, char_2)
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
# ## 5. Attention Visualization (XAI)
# The additive attention mechanism produces per-position weights
# showing which text regions the model considers most stylistically informative.

# %%
# Visualize attention for a few examples
with torch.no_grad():
    batch = next(iter(test_loader))
    char_1 = batch['char_ids_1'][:3].to(device)
    char_2 = batch['char_ids_2'][:3].to(device)

    _, _, (v1, v2, attn1, attn2) = model(char_1, char_2, return_embeddings=True)

    for i in range(min(3, len(attn1))):
        weights = attn1[i].cpu().numpy()
        print(f"\nSample {i+1} — Attention weights (top 10 positions):")
        top_pos = np.argsort(weights)[-10:][::-1]
        for pos in top_pos:
            print(f"  Position {pos}: weight={weights[pos]:.4f}")
