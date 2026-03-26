"""
COMP34812 — Solution 2 (Category B) Demo
Group 34 — AV Track
Adversarial Style-Content Disentanglement Network.
"""

# %% [markdown]
# # Solution 2 (Category B) — Demo / Inference
# ## Siamese Char-CNN + BiLSTM + GRL for Authorship Verification
#
# This notebook demonstrates inference with our Category B solution.
# The model uses a Siamese architecture with character-level CNN
# encoders, BiLSTM sequence modeling, additive attention, and a
# gradient reversal layer for adversarial topic debiasing.

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
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# %% [markdown]
# ## 1. Load Trained Model
#
# We instantiate the model with the exact architecture used during
# training (97-char vocab, 128 CNN filters, 128 BiLSTM hidden,
# 10 topic clusters) and load the trained weights.

# %%
model = AVCatBModel(
    vocab_size=VOCAB_SIZE, char_emb_dim=32,
    cnn_filters=128, lstm_hidden=128,
    proj_dim=128, num_topics=10,
).to(device)

model.load_state_dict(torch.load('models/av_cat_b_best.pt',
                                  map_location=device, weights_only=True))
model.eval()
print("Model loaded.")

# %% [markdown]
# ## 2. Load Test Data
#
# Text pairs are encoded at the character level (max 1500 chars).
# No augmentation is applied during inference.

# %%
test_df = load_av_data(split='dev')  # Replace with test data for submission
test_dataset = AVCharDataset(test_df, max_len=1500, augment=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print(f"Test data: {len(test_df)} pairs")

# %% [markdown]
# ## 3. Generate Predictions
#
# For each batch, we run the forward pass and apply sigmoid to get
# probabilities, then threshold at 0.5 for binary predictions.

# %%
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        char_1 = batch['char_ids_1'].to(device)
        char_2 = batch['char_ids_2'].to(device)
        logits, _ = model(char_1, char_2)
        probs = torch.sigmoid(logits.squeeze(-1))
        preds = (probs > 0.5).long()
        all_preds.extend(preds.cpu().numpy())

predictions = np.array(all_preds)
print(f"Predictions: {len(predictions)}")
print(f"Class distribution: {np.bincount(predictions)}")

# %% [markdown]
# ## 4. Save Predictions

# %%
save_predictions(predictions, 'predictions/Group_34_B.csv')
print("Saved to predictions/Group_34_B.csv")
