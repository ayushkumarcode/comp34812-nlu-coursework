"""AV Cat C — DeBERTa cross-encoder with max_len=256.

Shorter sequences: each text gets ~125 tokens, covers 60%
of texts. Faster training means more epochs and potentially
focuses on the most stylistically informative parts.
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
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
    def __init__(self, df, tokenizer, max_len=256):
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
    LR, BS, MAX_LEN = 2e-5, 16, 256
    EPOCHS, PATIENCE = 30, 7

    print(f"\n=== AV Cat C max_len=256 ===")
    print(f"LR={LR}, BS={BS}, MaxLen={MAX_LEN}")
    print(f"Epochs={EPOCHS}, Patience={PATIENCE}\n")

    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=False
    )

    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')

    train_ds = AVCEDataset(
        train_df, tok, max_len=MAX_LEN
    )
    dev_ds = AVCEDataset(
        dev_df, tok, max_len=MAX_LEN
    )
    dev_ds.labels = np.array(dev_labels, dtype=np.float32)

    train_loader = DataLoader(
        train_ds, batch_size=BS, shuffle=True,
        num_workers=4
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=BS, shuffle=False,
