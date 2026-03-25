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
