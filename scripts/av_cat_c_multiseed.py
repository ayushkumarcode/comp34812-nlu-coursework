"""AV Cat C — Multi-seed DeBERTa ensemble.

Trains the best config (R-Drop + label smoothing) with
3 different seeds, then ensembles probabilities.
Diversity from random seed -> better ensemble.
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
    sl = labels * (1 - s) + 0.5 * s
    return F.binary_cross_entropy_with_logits(logits, sl)


def train_one_seed(seed, train_df, dev_df, dev_labels,
                   tok, device, save_dir, pred_dir):
    from transformers import AutoModel

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    MN = 'microsoft/deberta-v3-base'
    ML, BS, LR = 384, 8, 2e-5
    EPOCHS, PAT = 25, 7
    ALPHA, SMOOTH = 0.5, 0.05

    train_ds = AVCEDataset(train_df, tok, max_len=ML)
    dev_ds = AVCEDataset(dev_df, tok, max_len=ML)
    dev_ds.labels = np.array(dev_labels, dtype=np.float32)

    tl = DataLoader(
        train_ds, batch_size=BS, shuffle=True,
        num_workers=4
    )
    dl = DataLoader(
        dev_ds, batch_size=BS, shuffle=False,
        num_workers=4
    )

    encoder = AutoModel.from_pretrained(MN).to(device)
    hs = encoder.config.hidden_size
    classifier = nn.Sequential(
        nn.Dropout(0.1), nn.Linear(hs, 256),
        nn.GELU(), nn.Dropout(0.2),
        nn.Linear(256, 1),
    ).to(device)

    all_params = (
        list(encoder.parameters())
        + list(classifier.parameters())
    )
    optimizer = AdamW([
        {'params': encoder.parameters(), 'lr': LR},
        {'params': classifier.parameters(), 'lr': 5e-4},
    ], weight_decay=0.01)
    total_steps = EPOCHS * len(tl)
    warmup = total_steps // 10
    sched = get_cosine_warmup(optimizer, warmup, total_steps)
    scaler = GradScaler('cuda')

    y_dev = np.array(dev_labels)
    best_f1, pat = 0, 0
    best_probs = None

    print(f"\n--- Seed {seed} ---")
    for ep in range(1, EPOCHS + 1):
        encoder.train()
        classifier.train()
        tl_sum, nb = 0, 0
        for batch in tl:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                o1 = encoder(input_ids=ids,
                             attention_mask=mask)
                l1 = classifier(
                    o1.last_hidden_state[:, 0]
                ).squeeze(-1)
                o2 = encoder(input_ids=ids,
                             attention_mask=mask)
                l2 = classifier(
                    o2.last_hidden_state[:, 0]
                ).squeeze(-1)
                loss1 = smooth_bce(l1, labels, SMOOTH)
