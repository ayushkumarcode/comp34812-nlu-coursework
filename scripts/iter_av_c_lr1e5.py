"""AV Cat C iter7 — Cross-Encoder DeBERTa LR=1e-5 (lower than current best 2e-5).

The current best AV Cat C (0.8074) used LR=2e-5, no R-Drop.
R-Drop with LR=3e-5 failed completely (model didn't learn).
Try LR=1e-5 with more epochs instead.
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


def main():
    from transformers import AutoTokenizer, AutoModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    MODEL_NAME = 'microsoft/deberta-v3-base'
    LR = 1e-5
    BATCH_SIZE = 8
    MAX_LEN = 384
    EPOCHS = 30
    PATIENCE = 10

    print(f"\n=== AV Cat C — Cross-Encoder LR={LR} ===")
    print(f"BS={BATCH_SIZE}, MaxLen={MAX_LEN}")
    print(f"Epochs={EPOCHS}, Patience={PATIENCE}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')

    train_dataset = AVCrossEncoderDataset(train_df, tokenizer, max_len=MAX_LEN)
    dev_dataset = AVCrossEncoderDataset(dev_df, tokenizer, max_len=MAX_LEN)
    dev_dataset.labels = np.array(dev_labels, dtype=np.float32)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
    classifier = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(encoder.config.hidden_size, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
    ).to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = AdamW([
        {'params': encoder.parameters(), 'lr': LR},
        {'params': classifier.parameters(), 'lr': 5e-4},
    ], weight_decay=0.01)
    scaler = GradScaler('cuda')

    best_f1 = 0
    patience_counter = 0
    save_dir = PROJECT_ROOT / 'models'
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        classifier.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = encoder(input_ids=ids, attention_mask=mask)
                cls_repr = outputs.last_hidden_state[:, 0, :]
                logits = classifier(cls_repr)
                loss = bce_loss(logits.squeeze(-1), labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(classifier.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        # Evaluate
        encoder.eval()
        classifier.eval()
        preds, probs_all, labels_all = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                outputs = encoder(input_ids=ids, attention_mask=mask)
                cls_repr = outputs.last_hidden_state[:, 0, :]
                logits = classifier(cls_repr).squeeze(-1)
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).long()
                preds.extend(pred.cpu().numpy())
                probs_all.extend(probs.cpu().numpy())
                labels_all.extend(batch['label'].numpy())

        dev_f1 = f1_score(labels_all, preds, average='macro', zero_division=0)
        print(f"Epoch {epoch:3d} | Loss: {total_loss/n_batches:.4f} | Dev F1: {dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save({
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict(),
            }, save_dir / 'av_cat_c_lr1e5_best.pt')
            np.save(save_dir / 'av_cat_c_lr1e5_probs.npy', np.array(probs_all))
            print(f"  -> Best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Threshold search
    print(f"\nBest AV Cat C (LR=1e-5) F1: {best_f1:.4f}")
    ckpt = torch.load(save_dir / 'av_cat_c_lr1e5_best.pt', weights_only=True)
    encoder.load_state_dict(ckpt['encoder'])
    classifier.load_state_dict(ckpt['classifier'])
    encoder.eval()
    classifier.eval()
