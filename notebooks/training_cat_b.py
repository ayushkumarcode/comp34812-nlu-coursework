"""
COMP34812 — Solution 2 (Category B) Training Notebook
Group 34

Neural architecture training for Authorship Verification / NLI.
- AV: Adversarial Style-Content Disentanglement Network (Siamese Char-CNN+BiLSTM+GRL)
- NLI: ESIM + KIM (BiLSTM + Cross-Attention + WordNet Knowledge)

To convert: jupyter nbconvert --to notebook training_cat_b.py
"""

# %% [markdown]
# # Solution 2 (Category B) — Training
# ## Adversarial Style-Content Disentanglement Network
#
# This notebook trains our Category B solution: a Siamese neural network
# with character-level CNN, BiLSTM, additive attention, gradient reversal
# for topic debiasing, and contrastive embedding loss.

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
# ## 1. Load and Prepare Data

# %%
train_df = load_av_data(split='train')
dev_df = load_av_data(split='dev')
dev_labels = load_solution_labels(task='av')
print(f"Train: {len(train_df)}, Dev: {len(dev_df)}")

# Generate topic pseudo-labels for adversarial training
all_texts = list(train_df['text_1']) + list(train_df['text_2'])
topic_labels = generate_topic_labels(all_texts, n_clusters=10)
train_topics = topic_labels[:len(train_df)]

# %% [markdown]
# ## 2. Create Datasets

# %%
from torch.utils.data import DataLoader

train_dataset = AVCharDataset(train_df, max_len=1500, augment=True, topic_labels=train_topics)
dev_dataset = AVCharDataset(dev_df, max_len=1500, augment=False)
dev_dataset.labels = np.array(dev_labels, dtype=np.float32)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False, num_workers=4)

# %% [markdown]
# ## 3. Build Model
#
# Architecture:
# - Character Embedding (32d) -> Multi-width Conv1D (3,5,7) -> MaxPool
# - BiLSTM (128 hidden) -> Additive Attention
# - Projection to 128d style embedding
# - Comparison: [v1, v2, |v1-v2|, v1*v2] -> MLP
# - GRL topic adversarial head for style-content disentanglement

# %%
num_topics = int(topic_labels.max()) + 1
model = AVCatBModel(
    vocab_size=VOCAB_SIZE, char_emb_dim=32,
    cnn_filters=128, lstm_hidden=128,
    proj_dim=128, num_topics=num_topics,
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## 4. Training Loop
# See src/training/train_av_cat_b.py for the full training procedure.
# Key training details:
# - Composite loss: BCE + 0.2*Contrastive + 0.1*Adversarial
# - AdamW optimizer, lr=1e-3, cosine annealing
# - GRL lambda ramps from 0 to 0.1 over first 5 epochs
# - Contrastive loss introduced at epoch 2
# - Character perturbation augmentation (5% per-char)
# - Random truncation (80-100%) during training
# - Early stopping with patience=7 on dev macro_f1

# %%
print("Training details in src/training/train_av_cat_b.py")
print("Run: python -m src.training.train_av_cat_b")
print("Or:  sbatch scripts/train_av_cat_b.sh (GPU cluster)")
