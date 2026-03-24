"""AV Cat C — Cross-Encoder DeBERTa with R-Drop regularization.

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

from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics


class AVCrossEncoderDataset(Dataset):
    """AV dataset for cross-encoder: tokenize text_1 + text_2 as pair."""
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


class AVCrossEncoder(nn.Module):
    """Simple cross-encoder: DeBERTa + classification head."""
    def __init__(self, model_name='microsoft/deberta-v3-base'):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_repr)
        return logits


def compute_rdrop_loss(logits1, logits2, labels, bce_loss, alpha=1.0):
    """Compute R-Drop loss: task loss + symmetric KL divergence."""
    loss1 = bce_loss(logits1, labels)
    loss2 = bce_loss(logits2, labels)
    task_loss = (loss1 + loss2) / 2

    # R-Drop KL divergence for binary classification
    p1 = torch.sigmoid(logits1)
    p2 = torch.sigmoid(logits2)
    dist1 = torch.stack([p1, 1 - p1], dim=-1).clamp(min=1e-7)
    dist2 = torch.stack([p2, 1 - p2], dim=-1).clamp(min=1e-7)
    kl_loss = (F.kl_div(dist1.log(), dist2, reduction='batchmean') +
               F.kl_div(dist2.log(), dist1, reduction='batchmean')) / 2

    total_loss = task_loss + alpha * kl_loss
    return total_loss, task_loss, kl_loss


def main():
    from transformers import AutoTokenizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    MODEL_NAME = 'microsoft/deberta-v3-base'
    LR = 2e-5
    BATCH_SIZE = 8
    MAX_LEN = 384
    EPOCHS = 25
    PATIENCE = 7
    ALPHA = 1.0

    print(f"\n=== AV Cat C + R-Drop (alpha={ALPHA}) ===")
    print(f"LR={LR}, BS={BATCH_SIZE}, MaxLen={MAX_LEN}")
    print(f"Epochs={EPOCHS}, Patience={PATIENCE}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')

    train_dataset = AVCrossEncoderDataset(
        train_df, tokenizer, max_len=MAX_LEN)
    dev_dataset = AVCrossEncoderDataset(
        dev_df, tokenizer, max_len=MAX_LEN)
    dev_dataset.labels = np.array(dev_labels, dtype=np.float32)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = AVCrossEncoder(model_name=MODEL_NAME).to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': LR},
        {'params': model.classifier.parameters(), 'lr': 5e-4},
    ], weight_decay=0.01)
    scaler = GradScaler('cuda')

    best_f1 = 0
    patience_counter = 0
    save_dir = PROJECT_ROOT / 'models'
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_task, total_kl = 0, 0, 0
        n_batches = 0

        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                # Two forward passes with different dropout masks (R-Drop)
                logits1 = model(ids, mask).squeeze(-1)
                logits2 = model(ids, mask).squeeze(-1)
                loss, task_loss, kl_loss = compute_rdrop_loss(
                    logits1, logits2, labels, bce_loss, alpha=ALPHA)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_task += task_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        # Evaluate
        model.eval()
        preds, labels_all = [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                logits = model(ids, mask).squeeze(-1)
                pred = (torch.sigmoid(logits) > 0.5).long()
                preds.extend(pred.cpu().numpy())
                labels_all.extend(batch['label'].numpy())

        dev_f1 = f1_score(labels_all, preds, average='macro', zero_division=0)
        print(f"Epoch {epoch:3d} | Loss: {total_loss/n_batches:.4f} "
              f"(task={total_task/n_batches:.4f}, "
              f"kl={total_kl/n_batches:.4f}) | Dev F1: {dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(),
                       save_dir / 'av_cat_c_rdrop_best.pt')
            print(f"  -> Best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
