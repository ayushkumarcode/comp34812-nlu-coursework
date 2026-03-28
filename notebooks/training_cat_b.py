"""
cat B training — siamese char-CNN + BiLSTM + GRL for AV (group 34).

to convert: python scripts/convert_to_ipynb.py notebooks/training_cat_b.py
"""

# %% [markdown]
# # Cat B — Training
# ## Adversarial Style-Content Disentanglement Network
#
# this trains our cat B solution: a siamese network with character-level
# CNN, BiLSTM, additive attention, and gradient reversal for topic
# debiasing. the final version (v3) uses BCE + topic adversarial CE only
# — we dropped contrastive loss since it didn't help.

# %%
# !pip install torch scikit-learn numpy pandas tqdm

# %%
import sys
import time
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.models.av_cat_b_model import AVCatBModel
from src.models.av_cat_b_dataset import AVCharDataset, generate_topic_labels, VOCAB_SIZE
from src.scorer import compute_all_metrics, print_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# %% [markdown]
# ## 1. Load and prepare data

# %%
train_df = load_av_data(split='train')
dev_df = load_av_data(split='dev')
dev_labels = load_solution_labels(task='av')
print(f"Train: {len(train_df)}, Dev: {len(dev_df)}")

# generate topic pseudo-labels for the adversarial head
all_texts = list(train_df['text_1']) + list(train_df['text_2'])
topic_labels = generate_topic_labels(all_texts, n_clusters=10)
train_topics = topic_labels[:len(train_df)]
num_topics = int(topic_labels.max()) + 1

# %% [markdown]
# ## 2. Create datasets

# %%
from torch.utils.data import DataLoader

train_dataset = AVCharDataset(train_df, max_len=1500, augment=True, topic_labels=train_topics)
dev_dataset = AVCharDataset(dev_df, max_len=1500, augment=False)
dev_dataset.labels = np.array(dev_labels, dtype=np.float32)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# %% [markdown]
# ## 3. Build model
#
# Architecture breakdown:
# - Char embedding (32d) -> multi-width Conv1D (3,5,7 with 128 filters each) -> maxpool
# - BiLSTM (128 hidden, bidirectional) -> additive attention
# - Project to 128d style embedding
# - Comparison: [v1, v2, |v1-v2|, v1*v2] = 512d -> MLP(512->256->64->1)
# - GRL topic adversarial head (128->64->num_topics)

# %%
model = AVCatBModel(
    vocab_size=VOCAB_SIZE, char_emb_dim=32,
    cnn_filters=128, lstm_hidden=128,
    proj_dim=128, num_topics=num_topics,
    grl_lambda=0.0,  # Starts at 0, ramps to 0.05
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## 4. Training Configuration
#
# Key training details (v3 — final version):
# - Loss: BCE + topic adversarial ONLY (NO contrastive loss)
# - AdamW optimizer, lr=2e-4, weight_decay=1e-4
# - CosineAnnealingWarmRestarts scheduler, T_0=30, T_mult=2
# - GRL lambda: linear ramp 0 -> 0.05 over epochs 1-20
# - Topic adversarial weight: 0.02, introduced from epoch 15
# - Character perturbation augmentation (5% per-char)
# - Random truncation (80-100%) during training
# - Early stopping with patience=20 on dev macro_f1
# - Gradient clipping: max_norm=5.0

# %%
print("Full training code: scripts/iter_av_b_v3.py")
print("Run on GPU: python scripts/iter_av_b_v3.py")
print("Or: sbatch scripts/train_av_cat_b.sh (CSF3 GPU cluster)")
