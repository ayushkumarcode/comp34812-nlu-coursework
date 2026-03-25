"""AV Cat B iter2 — Char-CNN+BiLSTM+GRL with lower LR and longer warmup.

Changes from v1:
- LR=5e-4 (was 1e-3) — slower, more stable
- BCE warmup=20 epochs (was 15)
- Contrastive from epoch 25 (was 15)
- Topic adversarial from epoch 15 (was 10)
- patience=15 (was 12)
- max_epochs=100 (was 80)
- Threshold search at end
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.av_cat_b_model import AVCatBModel
from src.models.av_cat_b_dataset import (
    AVCharDataset, generate_topic_labels, VOCAB_SIZE
)
from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics


def train_epoch(model, dataloader, optimizer, device, epoch,
                bce_loss_fn, contrastive_loss_fn,
                topic_loss_fn, contrastive_weight=0.2,
                topic_weight=0.1):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for batch in dataloader:
        char_1 = batch['char_ids_1'].to(device)
        char_2 = batch['char_ids_2'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        logits, topic_logits, (v1, v2, _, _) = model(
            char_1, char_2, return_embeddings=True)
        loss = bce_loss_fn(logits.squeeze(-1), labels)
        if contrastive_weight > 0:
            target = labels * 2 - 1
            loss = loss + contrastive_weight * contrastive_loss_fn(v1, v2, target)
        if topic_weight > 0 and 'topic' in batch:
            topic_labels = batch['topic'].to(device)
            loss = loss + topic_weight * topic_loss_fn(topic_logits, topic_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), f1_score(all_labels, all_preds, average='macro', zero_division=0)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            char_1 = batch['char_ids_1'].to(device)
            char_2 = batch['char_ids_2'].to(device)
            labels = batch['label']
            logits, _ = model(char_1, char_2)
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def main():
    print("=" * 60)
    print("  AV Cat B v2 — Lower LR, Longer Warmup")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n[1/5] Loading data...")
    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')

    print("\n[2/5] Generating topic labels...")
    all_texts = list(train_df['text_1']) + list(train_df['text_2'])
    topic_labels_all = generate_topic_labels(all_texts, n_clusters=10)
    train_topic = topic_labels_all[:len(train_df)]
    num_topics = int(topic_labels_all.max()) + 1

    print("\n[3/5] Creating datasets...")
    train_dataset = AVCharDataset(
        train_df, max_len=1500, augment=True, topic_labels=train_topic)
    dev_dataset = AVCharDataset(
        dev_df, max_len=1500, augment=False, topic_labels=None)
    dev_dataset.labels = np.array(dev_labels, dtype=np.float32)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                               num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

    print("\n[4/5] Building model...")
    model = AVCatBModel(
        vocab_size=VOCAB_SIZE, char_emb_dim=32, cnn_filters=128,
        lstm_hidden=128, proj_dim=128, num_topics=num_topics,
        grl_lambda=0.0).to(device)

    bce_loss_fn = nn.BCEWithLogitsLoss()
    contrastive_loss_fn = nn.CosineEmbeddingLoss(margin=0.3)
    topic_loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    print("\n[5/5] Training...")
    best_f1 = 0.0
    patience_counter = 0
    max_epochs = 100
    patience = 15
    save_dir = PROJECT_ROOT / 'models'
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        # Ramp GRL lambda slowly
        if epoch <= 15:
            grl_lambda = 0.05 * epoch / 15
        else:
            grl_lambda = 0.05
        model.grl.lambda_val = grl_lambda

        # Longer warmup: BCE-only for 20 epochs
        c_weight = 0.05 if epoch >= 25 else 0.0
        t_weight = 0.02 if epoch >= 15 else 0.0
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, device, epoch,
            bce_loss_fn, contrastive_loss_fn, topic_loss_fn,
            contrastive_weight=c_weight, topic_weight=t_weight)
        scheduler.step()

        dev_preds, dev_probs, dev_true = evaluate(model, dev_loader, device)
        dev_f1 = f1_score(dev_true, dev_preds, average='macro', zero_division=0)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
              f"Train F1: {train_f1:.4f} | Dev F1: {dev_f1:.4f} | "
              f"GRL: {grl_lambda:.3f} | Time: {elapsed:.1f}s")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / 'av_cat_b_v2_best.pt')
            np.save(save_dir / 'av_cat_b_v2_probs.npy', dev_probs)
            print(f"  -> New best (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final eval + threshold search
    print(f"\nBest AV Cat B v2 F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(save_dir / 'av_cat_b_v2_best.pt', weights_only=True))
    dev_preds, dev_probs, dev_true = evaluate(model, dev_loader, device)

    print("\nThreshold search:")
    best_thresh, best_thresh_f1 = 0.5, 0
    for t in np.arange(0.30, 0.72, 0.02):
        preds_t = (dev_probs >= t).astype(int)
        f1_t = f1_score(dev_true, preds_t, average='macro', zero_division=0)
        if f1_t > best_thresh_f1:
            best_thresh, best_thresh_f1 = t, f1_t
        print(f"  thresh={t:.2f}: F1={f1_t:.4f}")

    print(f"\nBest threshold: {best_thresh:.2f} -> F1={best_thresh_f1:.4f}")
    final_preds = (dev_probs >= best_thresh).astype(int)
    metrics = compute_all_metrics(dev_true, final_preds)
    print_metrics(metrics, "AV Cat B v2 — Final Dev Results")

    save_predictions(final_preds, PROJECT_ROOT / 'predictions' / 'av_Group_34_B_v2.csv')

    baselines = {'SVM': 0.5610, 'LSTM': 0.6226, 'BERT': 0.7854}
    for name, bl in baselines.items():
        gap = metrics['macro_f1'] - bl
        print(f"  vs {name} ({bl:.4f}): {'BEATS' if gap > 0 else 'BELOW'} by {gap:+.4f}")
    print("Done!")
