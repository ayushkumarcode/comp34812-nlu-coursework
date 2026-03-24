"""NLI Cat C — DeBERTa Cross-Encoder with R-Drop regularization.

R-Drop (Liang et al., 2021): Regularized Dropout for neural networks.
Two forward passes with different dropout masks, then minimize KL divergence
between their output distributions to improve consistency.

Total loss = (BCE1 + BCE2)/2 + alpha * symmetric_KL(p1, p2)
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics


class NLIDeBERTaDataset(Dataset):
    """Dataset for NLI cross-encoder DeBERTa."""

    def __init__(self, df, tokenizer, max_len=128):
        self.premises = list(df['premise'])
        self.hypotheses = list(df['hypothesis'])
        self.labels = df['label'].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.premises[idx], self.hypotheses[idx],
            truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float),
        }


class NLIDeBERTaCrossEncoder(nn.Module):
    """Cross-encoder DeBERTa for NLI (no adversarial head — clean version)."""

    def __init__(self, model_name='microsoft/deberta-v3-base'):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_repr)
        return logits


def compute_rdrop_loss(logits1, logits2, labels, bce_loss, alpha=1.0):
    """Compute R-Drop loss: task loss + symmetric KL divergence.

    Args:
        logits1: First forward pass logits (batch,)
        logits2: Second forward pass logits (batch,)
        labels: Ground truth labels (batch,)
        bce_loss: BCEWithLogitsLoss instance
        alpha: Weight for KL divergence term

    Returns:
        Total loss, task_loss, kl_loss
    """
    loss1 = bce_loss(logits1, labels)
    loss2 = bce_loss(logits2, labels)
    task_loss = (loss1 + loss2) / 2

    # R-Drop KL divergence for binary classification
    p1 = torch.sigmoid(logits1)
    p2 = torch.sigmoid(logits2)
    # Stack to get [p, 1-p] distributions
    dist1 = torch.stack([p1, 1 - p1], dim=-1).clamp(min=1e-7)
    dist2 = torch.stack([p2, 1 - p2], dim=-1).clamp(min=1e-7)
    # Symmetric KL divergence
    kl_loss = (F.kl_div(dist1.log(), dist2, reduction='batchmean') +
               F.kl_div(dist2.log(), dist1, reduction='batchmean')) / 2

    total_loss = task_loss + alpha * kl_loss
    return total_loss, task_loss, kl_loss


def main():
    from transformers import AutoTokenizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
