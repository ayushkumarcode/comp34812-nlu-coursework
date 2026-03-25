"""AV Cat C — DeBERTa with R-Drop + Label Smoothing.

Combines two regularization techniques:
1. R-Drop: consistency between two forward passes
2. Label Smoothing: soft targets (0.05) to prevent
   overconfident predictions
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

from src.data_utils import (
    load_av_data, load_solution_labels, save_predictions
)
from src.scorer import compute_all_metrics, print_metrics


class AVCEDataset(Dataset):
    """AV cross-encoder dataset."""
    def __init__(self, df, tokenizer, max_len=384):
        self.t1 = list(df['text_1'])
        self.t2 = list(df['text_2'])
        self.labels = df['label'].values.astype(np.float32)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tok(
            self.t1[idx], self.t2[idx],
            truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(
                self.labels[idx], dtype=torch.float
            ),
        }


class AVCE(nn.Module):
    """DeBERTa cross-encoder for AV."""
    def __init__(self, mn='microsoft/deberta-v3-base'):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(mn)
        hs = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(hs, 256),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.classifier(out.last_hidden_state[:, 0])


def smooth_bce(logits, labels, smoothing=0.05):
    """BCE loss with label smoothing."""
    smooth_labels = labels * (1 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(
        logits, smooth_labels
    )


def rdrop_loss_smooth(
    logits1, logits2, labels, alpha=1.0,
    smoothing=0.05
):
    """R-Drop loss with label smoothing."""
    loss1 = smooth_bce(logits1, labels, smoothing)
    loss2 = smooth_bce(logits2, labels, smoothing)
    task = (loss1 + loss2) / 2

    p1 = torch.sigmoid(logits1)
    p2 = torch.sigmoid(logits2)
    d1 = torch.stack([p1, 1 - p1], dim=-1).clamp(1e-7)
    d2 = torch.stack([p2, 1 - p2], dim=-1).clamp(1e-7)
    kl = (
        F.kl_div(d1.log(), d2, reduction='batchmean')
        + F.kl_div(d2.log(), d1, reduction='batchmean')
    ) / 2
    return task + alpha * kl, task, kl


def main():
    from transformers import AutoTokenizer

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")

