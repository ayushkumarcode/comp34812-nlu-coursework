"""
Cat B training script. BCE + topic adversarial loss, no contrastive.
AdamW lr=2e-4, cosine scheduler, GRL lambda ramps over first 20 epochs.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.av_cat_b_model import AVCatBModel
from src.models.av_cat_b_dataset import (
    AVCharDataset, generate_topic_labels, VOCAB_SIZE
)
from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics


def train_epoch(model, dataloader, optimizer, device,
                bce_loss_fn, topic_loss_fn, topic_weight=0.02):
    """One training epoch. Returns (avg_loss, macro_f1)."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        char_1 = batch['char_ids_1'].to(device)
        char_2 = batch['char_ids_2'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        logits, topic_logits, _ = model(
            char_1, char_2, return_embeddings=True
        )

        loss = bce_loss_fn(logits.squeeze(-1), labels)

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

    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, macro_f1


def evaluate(model, dataloader, device):
    """Run eval, return (preds, probs, labels)."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

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
    print("  AV Category B — Training (v3, final)")
    print("  BCE + Topic Adversarial, lr=2e-4, no contrastive")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("\n[1/5] Loading data...")
    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')
    print(f"  Train: {len(train_df)}, Dev: {len(dev_df)}")

    # Generate topic pseudo-labels
    print("\n[2/5] Generating topic labels...")
    all_texts = list(train_df['text_1']) + list(train_df['text_2'])
    topic_labels_all = generate_topic_labels(all_texts, n_clusters=10)
    train_topic = topic_labels_all[:len(train_df)]
    num_topics = int(topic_labels_all.max()) + 1

    # Create datasets
    print("\n[3/5] Creating datasets...")
    train_dataset = AVCharDataset(
        train_df, max_len=1500, augment=True, topic_labels=train_topic
    )
    dev_dataset = AVCharDataset(
        dev_df, max_len=1500, augment=False, topic_labels=None
    )
    dev_dataset.labels = np.array(dev_labels, dtype=np.float32)

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Build model
    print("\n[4/5] Building model...")
    model = AVCatBModel(
        vocab_size=VOCAB_SIZE,
        char_emb_dim=32,
        cnn_filters=128,
        lstm_hidden=128,
        proj_dim=128,
        num_topics=num_topics,
        grl_lambda=0.0,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Loss functions (NO contrastive loss in v3)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    topic_loss_fn = nn.CrossEntropyLoss()

    # Optimizer and scheduler (v3: lr=2e-4, T_0=30)
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    # Training loop
    print("\n[5/5] Training...")
    best_f1 = 0.0
    patience_counter = 0
    max_epochs = 120
    patience = 20
    save_dir = PROJECT_ROOT / 'models'
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # GRL lambda: ramp 0 -> 0.05 over epochs 1-20
        if epoch <= 20:
            grl_lambda = 0.05 * epoch / 20
        else:
            grl_lambda = 0.05
        model.grl.lambda_val = grl_lambda

        # Topic weight: 0.02 from epoch 15 onward
        t_weight = 0.02 if epoch >= 15 else 0.0

        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, device,
            bce_loss_fn, topic_loss_fn,
            topic_weight=t_weight,
        )
        scheduler.step()

        # Evaluate
        dev_preds, dev_probs, dev_true = evaluate(model, dev_loader, device)
        dev_f1 = f1_score(dev_true, dev_preds, average='macro', zero_division=0)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
              f"Train F1: {train_f1:.4f} | Dev F1: {dev_f1:.4f} | "
              f"GRL: {grl_lambda:.3f} | Time: {elapsed:.1f}s")

        # Save best model
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / 'av_cat_b_best.pt')
            print(f"  -> New best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model and final evaluation
    print("\n" + "=" * 60)
    print(f"  Best dev macro_f1: {best_f1:.4f}")
    print("=" * 60)

    model.load_state_dict(
        torch.load(save_dir / 'av_cat_b_best.pt', weights_only=True)
    )
    dev_preds, dev_probs, dev_true = evaluate(model, dev_loader, device)
    metrics = compute_all_metrics(dev_true, dev_preds)
    print_metrics(metrics, "AV Cat B — Final Dev Results")

    # Save predictions
    pred_path = PROJECT_ROOT / 'predictions' / 'av_Group_34_B.csv'
    pred_path.parent.mkdir(exist_ok=True)
    save_predictions(dev_preds, pred_path)

    # Baseline comparison
    baselines = {'SVM': 0.5610, 'LSTM': 0.6226, 'BERT': 0.7854}
    f1 = metrics['macro_f1']
    for name, baseline_f1 in baselines.items():
        gap = f1 - baseline_f1
        status = "BEATS" if gap > 0 else "BELOW"
        print(f"  vs {name} ({baseline_f1:.4f}): {status} by {gap:+.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
