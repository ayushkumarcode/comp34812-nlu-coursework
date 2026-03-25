"""AV Cat A — MLP with R-Drop on handcrafted features.

R-Drop (Liang et al., 2021) regularizes by doing two
forward passes with different dropout masks and minimizing
the KL divergence between their predictions.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (
    load_av_data, load_solution_labels, save_predictions
)
from src.av_pipeline import AVFeatureExtractor
from src.scorer import compute_all_metrics, print_metrics

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"Device: {device}")


class AV_MLP(nn.Module):
    """3-layer MLP for AV binary classification."""
    def __init__(self, input_dim, hidden_dims=(512, 256),
                 dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def rdrop_loss(logits1, logits2, labels, alpha=1.0):
    """R-Drop loss for binary classification."""
    bce = nn.BCEWithLogitsLoss()
    loss1 = bce(logits1, labels)
    loss2 = bce(logits2, labels)
    task_loss = (loss1 + loss2) / 2
    p1 = torch.sigmoid(logits1)
    p2 = torch.sigmoid(logits2)
    dist1 = torch.stack([p1, 1 - p1], dim=-1)
    dist2 = torch.stack([p2, 1 - p2], dim=-1)
    dist1 = dist1.clamp(min=1e-7)
    dist2 = dist2.clamp(min=1e-7)
    kl = (
        F.kl_div(dist1.log(), dist2, reduction='batchmean')
        + F.kl_div(dist2.log(), dist1, reduction='batchmean')
    ) / 2
    return task_loss + alpha * kl, task_loss, kl


def main():
    print("=== AV Cat A: MLP + R-Drop ===\n")

    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    y_train = train_df['label'].values.astype(np.float32)
    y_dev = np.array(
        load_solution_labels(task='av'), dtype=np.float32
    )

    print("Extracting features...")
    ext = AVFeatureExtractor(
        use_spacy=True, n_svd_components=100
    )
    ext.fit(train_df)
    X_train, fnames = ext.transform(train_df)
    X_dev, _ = ext.transform(dev_df)
    print(f"Features: {X_train.shape[1]}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_dev = scaler.transform(X_dev).astype(np.float32)

    X_tr = torch.tensor(X_train).to(device)
    y_tr = torch.tensor(y_train).to(device)
    X_dv = torch.tensor(X_dev).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True
    )

    configs = [
        {'hidden': (512, 256), 'drop': 0.3,
         'lr': 1e-3, 'alpha': 1.0},
        {'hidden': (512, 256, 128), 'drop': 0.3,
         'lr': 1e-3, 'alpha': 1.0},
        {'hidden': (512, 256), 'drop': 0.4,
         'lr': 1e-3, 'alpha': 0.5},
