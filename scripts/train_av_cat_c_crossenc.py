"""AV Cat C — Cross-Encoder DeBERTa (instead of Siamese)."""
import sys
import time
import argparse
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


def main():
    from transformers import AutoTokenizer, AutoModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    MODEL_NAME = 'microsoft/deberta-v3-base'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')

    MAX_LEN = 384
    BATCH_SIZE = 8
    EPOCHS = 25
    PATIENCE = 7
    LR = 2e-5

    train_dataset = AVCrossEncoderDataset(train_df, tokenizer, max_len=MAX_LEN)
    dev_dataset = AVCrossEncoderDataset(dev_df, tokenizer, max_len=MAX_LEN)
    dev_dataset.labels = np.array(dev_labels, dtype=np.float32)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Simple cross-encoder: DeBERTa + classification head
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
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(classifier.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        # Evaluate
        encoder.eval()
        classifier.eval()
        preds, labels_all = [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                outputs = encoder(input_ids=ids, attention_mask=mask)
                cls_repr = outputs.last_hidden_state[:, 0, :]
                logits = classifier(cls_repr)
                pred = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
                preds.extend(pred.cpu().numpy())
                labels_all.extend(batch['label'].numpy())

        dev_f1 = f1_score(labels_all, preds, average='macro', zero_division=0)
        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_loader):.4f} | Dev F1: {dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
            }, save_dir / 'av_cat_c_crossenc_best.pt')
            print(f"  -> Best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation
    print(f"\nBest AV Cat C (cross-encoder) dev F1: {best_f1:.4f}")
    checkpoint = torch.load(save_dir / 'av_cat_c_crossenc_best.pt', weights_only=True)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    encoder.eval()
    classifier.eval()

    final_preds = []
    with torch.no_grad():
        for batch in dev_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            outputs = encoder(input_ids=ids, attention_mask=mask)
            cls_repr = outputs.last_hidden_state[:, 0, :]
            logits = classifier(cls_repr)
            pred = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
            final_preds.extend(pred.cpu().numpy())

    y_dev = np.array(dev_labels)
    final_metrics = compute_all_metrics(y_dev, np.array(final_preds))
    print_metrics(final_metrics, "AV Cat C (Cross-Encoder) — Final Dev Results")

    pred_path = PROJECT_ROOT / 'predictions' / 'av_Group_34_C_crossenc.csv'
    pred_path.parent.mkdir(exist_ok=True)
    save_predictions(final_preds, pred_path)
    print(f"Predictions saved to {pred_path}")

    baselines = {'SVM': 0.5610, 'LSTM': 0.6226, 'BERT': 0.7854}
    for name, baseline_f1 in baselines.items():
        gap = final_metrics['macro_f1'] - baseline_f1
        status = "BEATS" if gap > 0 else "BELOW"
        print(f"  vs {name} ({baseline_f1:.4f}): {status} by {gap:+.4f}")
    print("Done!")


if __name__ == '__main__':
    main()
