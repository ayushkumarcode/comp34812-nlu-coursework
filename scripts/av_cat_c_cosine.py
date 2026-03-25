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
    LR, BS, MAX_LEN = 2e-5, 8, 384
    EPOCHS, PATIENCE = 30, 10
    T_0, T_MULT = 3, 2

    print(f"\n=== AV Cat C + CosineAnnealingWarmRestarts ===")
    print(f"LR={LR}, BS={BS}, MaxLen={MAX_LEN}")
    print(f"T_0={T_0}, T_mult={T_MULT}")
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
        num_workers=4
    )

    model = AVCE(mn=MODEL_NAME).to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': LR},
        {'params': model.classifier.parameters(),
         'lr': 5e-4},
    ], weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_MULT, eta_min=1e-7
    )
    scaler = GradScaler('cuda')

    best_f1, pat = 0, 0
    save_dir = PROJECT_ROOT / 'models'
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n_b = 0, 0
        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(ids, mask).squeeze(-1)
                loss = bce(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_b += 1

        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        model.eval()
        preds, labels_all = [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                logits = model(ids, mask).squeeze(-1)
                p = (torch.sigmoid(logits) > 0.5).long()
                preds.extend(p.cpu().numpy())
                labels_all.extend(batch['label'].numpy())

        dev_f1 = f1_score(
            labels_all, preds, average='macro',
            zero_division=0
        )
        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {total_loss/n_b:.4f} | "
            f"LR: {cur_lr:.2e} | "
            f"Dev F1: {dev_f1:.4f}"
        )

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            pat = 0
            torch.save(
                model.state_dict(),
                save_dir / 'av_cat_c_cosine_best.pt'
            )
            print(f"  -> Best (F1={best_f1:.4f})")
        else:
            pat += 1
            if pat >= PATIENCE:
                print(f"Early stop at epoch {epoch}")
                break

    # Final eval + threshold sweep
    print(f"\nBest F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(
        save_dir / 'av_cat_c_cosine_best.pt',
        weights_only=True
    ))
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dev_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(ids, mask).squeeze(-1)
            all_probs.extend(
                torch.sigmoid(logits).cpu().numpy()
            )
    all_probs = np.array(all_probs)
    y_dev = np.array(dev_labels)

    bf, bt = 0, 0.5
    for t in np.arange(0.30, 0.70, 0.005):
        f1 = f1_score(
            y_dev, (all_probs > t).astype(int),
            average='macro'
        )
        if f1 > bf:
            bf = f1
            bt = t
    final_preds = (all_probs > bt).astype(int)
    print(f"Threshold: F1={bf:.4f} t={bt:.3f}")
