"""AV Cat C iter6 — Cross-Encoder DeBERTa with R-Drop + LR=3e-5.

Changes from best (crossenc, LR=2e-5, no R-Drop):
- Add R-Drop regularization (alpha=1.0)
- LR=3e-5 for encoder (slightly faster learning)
- Epochs=20, patience=8
- Threshold search at end
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

from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics


class AVCrossEncoderDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=384):
        self.texts_1 = list(df['text_1'])
        self.texts_2 = list(df['text_2'])
        self.labels = df['label'].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts_1[idx], self.texts_2[idx],
            truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float),
        }


def compute_rdrop_loss(logits1, logits2, labels, bce_fn, alpha=1.0):
    loss1 = bce_fn(logits1, labels)
    loss2 = bce_fn(logits2, labels)
    task_loss = (loss1 + loss2) / 2
    p1 = torch.sigmoid(logits1)
    p2 = torch.sigmoid(logits2)
    dist1 = torch.stack([p1, 1 - p1], dim=-1).clamp(min=1e-7)
    dist2 = torch.stack([p2, 1 - p2], dim=-1).clamp(min=1e-7)
    kl = (F.kl_div(dist1.log(), dist2, reduction='batchmean') +
          F.kl_div(dist2.log(), dist1, reduction='batchmean')) / 2
    return task_loss + alpha * kl, task_loss, kl


def main():
    from transformers import AutoTokenizer, AutoModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    MODEL_NAME = 'microsoft/deberta-v3-base'
    LR = 3e-5
    BATCH_SIZE = 8
    MAX_LEN = 384
    EPOCHS = 20
    PATIENCE = 8
    ALPHA = 1.0

    print(f"\n=== AV Cat C — Cross-Encoder + R-Drop ===")
    print(f"LR={LR}, BS={BATCH_SIZE}, MaxLen={MAX_LEN}")
    print(f"Epochs={EPOCHS}, Patience={PATIENCE}, Alpha={ALPHA}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

