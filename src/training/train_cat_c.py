"""
Category C — DeBERTa Training Script.
Supports both AV (Siamese) and NLI (cross-encoder) modes.
Run on CSF3 with GPU: sbatch train_cat_c.sh
"""

import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_av_data, load_nli_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics


class AVDeBERTaDataset(Dataset):
    """Dataset for AV Siamese DeBERTa."""

    def __init__(self, df, tokenizer, max_len=256):
        self.texts_1 = list(df['text_1'])
        self.texts_2 = list(df['text_2'])
        self.labels = df['label'].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc1 = self.tokenizer(
            self.texts_1[idx], truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt'
        )
        enc2 = self.tokenizer(
            self.texts_2[idx], truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids_1': enc1['input_ids'].squeeze(0),
            'attention_mask_1': enc1['attention_mask'].squeeze(0),
            'input_ids_2': enc2['input_ids'].squeeze(0),
            'attention_mask_2': enc2['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float),
        }


class NLIDeBERTaDataset(Dataset):
    """Dataset for NLI cross-encoder DeBERTa."""

    def __init__(self, df, tokenizer, max_len=128, hyp_max_len=48):
        self.premises = list(df['premise'])
        self.hypotheses = list(df['hypothesis'])
        self.labels = df['label'].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.hyp_max_len = hyp_max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Cross-encoder: premise + hypothesis
        enc = self.tokenizer(
            self.premises[idx], self.hypotheses[idx],
            truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt'
        )
        # Hypothesis-only (for adversarial debiasing)
        hyp_enc = self.tokenizer(
            self.hypotheses[idx], truncation=True, max_length=self.hyp_max_len,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'hyp_input_ids': hyp_enc['input_ids'].squeeze(0),
            'hyp_attention_mask': hyp_enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float),
        }


def train_av(args):
    """Train AV Cat C — Siamese DeBERTa."""
    from transformers import AutoTokenizer
    from src.models.cat_c_deberta import AVDeBERTaSiamese

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load data
    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')

    train_dataset = AVDeBERTaDataset(train_df, tokenizer, max_len=256)
    dev_dataset = AVDeBERTaDataset(dev_df, tokenizer, max_len=256)
    dev_dataset.labels = np.array(dev_labels, dtype=np.float32)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = AVDeBERTaSiamese(model_name=args.model_name).to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    contrastive_loss = nn.CosineEmbeddingLoss(margin=0.3)

    # Optimizer with discriminative LR
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': 2e-5},
        {'params': model.scalar_mix.parameters(), 'lr': 1e-3},
        {'params': model.style_proj.parameters(), 'lr': 5e-4},
        {'params': model.classifier.parameters(), 'lr': 5e-4},
        {'params': model.topic_head.parameters(), 'lr': 5e-4},
    ]
    optimizer = AdamW(param_groups, weight_decay=0.01)
    scaler = GradScaler()

    best_f1 = 0
    patience_counter = 0
    save_dir = PROJECT_ROOT / 'models'
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            ids1 = batch['input_ids_1'].to(device)
            mask1 = batch['attention_mask_1'].to(device)
            ids2 = batch['input_ids_2'].to(device)
            mask2 = batch['attention_mask_2'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with autocast():
                logits, topic_logits, (v1, v2) = model(ids1, mask1, ids2, mask2)
                loss_bce = bce_loss(logits.squeeze(-1), labels)
                target = labels * 2 - 1
                loss_con = contrastive_loss(v1, v2, target)
                loss = loss_bce + 0.3 * loss_con

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        preds, labels_all = [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids1 = batch['input_ids_1'].to(device)
                mask1 = batch['attention_mask_1'].to(device)
                ids2 = batch['input_ids_2'].to(device)
                mask2 = batch['attention_mask_2'].to(device)
                logits, _, _ = model(ids1, mask1, ids2, mask2)
                pred = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
                preds.extend(pred.cpu().numpy())
                labels_all.extend(batch['label'].numpy())

        dev_f1 = f1_score(labels_all, preds, average='macro', zero_division=0)
        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_loader):.4f} | Dev F1: {dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / 'av_cat_c_best.pt')
            print(f"  -> Best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest AV Cat C dev F1: {best_f1:.4f}")


def train_nli(args):
    """Train NLI Cat C — Cross-Encoder DeBERTa."""
    from transformers import AutoTokenizer
    from src.models.cat_c_deberta import NLIDeBERTaCrossEncoder

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_df = load_nli_data(split='train')
    dev_df = load_nli_data(split='dev')
    dev_labels = load_solution_labels(task='nli')

    train_dataset = NLIDeBERTaDataset(train_df, tokenizer)
    dev_dataset = NLIDeBERTaDataset(dev_df, tokenizer)
    dev_dataset.labels = np.array(dev_labels, dtype=np.float32)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = NLIDeBERTaCrossEncoder(model_name=args.model_name).to(device)
    bce_loss = nn.BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = GradScaler()

    best_f1 = 0
    patience_counter = 0
    save_dir = PROJECT_ROOT / 'models'
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            hyp_ids = batch['hyp_input_ids'].to(device)
            hyp_mask = batch['hyp_attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with autocast():
                logits, adv_logits = model(ids, mask, hyp_ids, hyp_mask)
                loss = bce_loss(logits.squeeze(-1), labels)
                if adv_logits is not None:
                    loss_adv = bce_loss(adv_logits.squeeze(-1), labels)
                    loss = loss + 0.1 * loss_adv

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        preds, labels_all = [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                logits, _ = model(ids, mask)
                pred = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
                preds.extend(pred.cpu().numpy())
                labels_all.extend(batch['label'].numpy())

        dev_f1 = f1_score(labels_all, preds, average='macro', zero_division=0)
        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_loader):.4f} | Dev F1: {dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / 'nli_cat_c_best.pt')
            print(f"  -> Best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest NLI Cat C dev F1: {best_f1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['av', 'nli'], required=True)
    parser.add_argument('--model_name', default='microsoft/deberta-v3-base')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    if args.task == 'av':
        train_av(args)
    else:
        train_nli(args)


if __name__ == '__main__':
    main()
