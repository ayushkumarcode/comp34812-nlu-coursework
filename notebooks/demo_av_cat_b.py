"""cat B demo — siamese char-CNN + BiLSTM + GRL for AV (group 34)"""

# %% [markdown]
# # Cat B — Demo / Inference
# ## Siamese Char-CNN + BiLSTM + GRL
#
# this notebook runs inference with our cat B solution. it's a siamese
# architecture that uses character-level CNN encoders, BiLSTM for
# sequence modeling, additive attention, and a gradient reversal layer
# to debias for topic.

# %%
# !pip install torch numpy pandas

# %%
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_av_data, clean_text, save_predictions
from src.models.av_cat_b_model import AVCatBModel
from src.models.av_cat_b_dataset import char_encode, VOCAB_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# set this to a csv path for custom inference, or leave as None for dev
INPUT_FILE = None  # e.g. 'test_data_av.csv'

# %% [markdown]
# ## 1. Load the trained model
#
# We set up the model with the same architecture used during training
# (97-char vocab, 128 CNN filters, 128 BiLSTM hidden, 10 topic
# clusters) and load the saved weights.

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
# Load data: use INPUT_FILE if set, otherwise default to dev split.
if INPUT_FILE is not None:
    test_df = pd.read_csv(INPUT_FILE, quotechar='"', engine='python')
    test_df['text_1'] = test_df['text_1'].apply(
        lambda x: clean_text(x, lowercase=False))
    test_df['text_2'] = test_df['text_2'].apply(
        lambda x: clean_text(x, lowercase=False))
else:
    test_df = load_av_data(split='dev')
print(f"Test data: {len(test_df)} pairs")

# Encode characters directly (no label column required)
max_len = 1500
encoded_1 = [char_encode(t, max_len) for t in test_df['text_1']]
encoded_2 = [char_encode(t, max_len) for t in test_df['text_2']]
ids_1 = torch.tensor(np.array(encoded_1), dtype=torch.long)
ids_2 = torch.tensor(np.array(encoded_2), dtype=torch.long)

# %% [markdown]
# ## 3. Generate Predictions
#
# For each batch, we run the forward pass and apply sigmoid to get
# probabilities, then threshold at 0.5 for binary predictions.

# %%
batch_size = 64
all_preds = []
with torch.no_grad():
    for start in range(0, len(test_df), batch_size):
        end = min(start + batch_size, len(test_df))
        b1 = ids_1[start:end].to(device)
        b2 = ids_2[start:end].to(device)
        logits, _ = model(b1, b2)
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
