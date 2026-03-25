"""AV Cat C — R-Drop + Label Smoothing + Long Training.

The previous rdrop_smooth run was still improving at epoch
25, so train for 40 epochs with patience 10. Uses the
identical architecture and hyperparameters (alpha=0.5,
smoothing=0.05, LR=2e-5) that achieved 0.8247.
"""
import sys
import math
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


def get_cosine_warmup(opt, warmup, total):
    def fn(step):
        if step < warmup:
            return float(step) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def smooth_bce(logits, labels, s=0.05):
    sl = labels * (1 - s) + (1 - labels) * s
    return F.binary_cross_entropy_with_logits(logits, sl)


def main():
    from transformers import AutoTokenizer, AutoModel

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    MN = 'microsoft/deberta-v3-base'
    ML, BS, LR = 384, 8, 2e-5
    EPOCHS, PAT = 40, 10
    ALPHA, SMOOTH = 0.5, 0.05

    print(f"\n=== AV Cat C R-Drop+Smooth LONG ===")
    print(f"Ep={EPOCHS} Pat={PAT} alpha={ALPHA}")
    print(f"Previous best: 0.8247 at ep 25\n")
