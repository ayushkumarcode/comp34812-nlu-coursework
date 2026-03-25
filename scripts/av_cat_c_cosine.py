"""AV Cat C — DeBERTa with CosineAnnealingWarmRestarts.

Current best uses linear LR decay. Cosine annealing with
warm restarts (T_0=3, T_mult=2) can help escape local
minima by periodically resetting learning rate.
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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


def main():
    from transformers import AutoTokenizer

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    MODEL_NAME = 'microsoft/deberta-v3-base'
