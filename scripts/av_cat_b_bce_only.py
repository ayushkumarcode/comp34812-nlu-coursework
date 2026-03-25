"""AV Cat B — BCE-only training (no contrastive, no GRL).

Tests whether the adversarial/contrastive losses actually
help or hurt. Uses same char-CNN+BiLSTM architecture but
trains with pure BCE loss throughout.
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
from src.data_utils import (
    load_av_data, load_solution_labels, save_predictions
)
from src.scorer import compute_all_metrics, print_metrics


def evaluate(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            c1 = batch['char_ids_1'].to(device)
            c2 = batch['char_ids_2'].to(device)
            labels = batch['label']
            logits, _ = model(c1, c2)
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return (np.array(all_preds),
            np.array(all_probs),
            np.array(all_labels))


def main():
    print("=" * 60)
    print("  AV Cat B — BCE-Only Training")
    print("=" * 60)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')

    # No topic labels needed for BCE-only
    train_dataset = AVCharDataset(
        train_df, max_len=1500, augment=True,
        topic_labels=None
    )
    dev_dataset = AVCharDataset(
        dev_df, max_len=1500, augment=False,
        topic_labels=None
    )
    dev_dataset.labels = np.array(
        dev_labels, dtype=np.float32
    )

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True
    )

    model = AVCatBModel(
        vocab_size=VOCAB_SIZE, char_emb_dim=32,
        cnn_filters=128, lstm_hidden=128,
        proj_dim=128, num_topics=10,
        grl_lambda=0.0,
    ).to(device)
    total_params = sum(
        p.numel() for p in model.parameters()
    )
    print(f"  Total parameters: {total_params:,}")

    bce_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(), lr=2e-4, weight_decay=1e-4
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_f1, patience_counter = 0.0, 0
    max_epochs, patience = 100, 15
    save_dir = PROJECT_ROOT / 'models'
    save_dir.mkdir(exist_ok=True)

    print("\nTraining with BCE-only loss...")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0

        for batch in train_loader:
            c1 = batch['char_ids_1'].to(device)
            c2 = batch['char_ids_2'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits, _ = model(c1, c2)
            loss = bce_loss_fn(
                logits.squeeze(-1), labels
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        preds, probs, true = evaluate(
            model, dev_loader, device
        )
        dev_f1 = f1_score(
            true, preds, average='macro', zero_division=0
        )
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {total_loss/len(train_loader):.4f} | "
            f"Dev F1: {dev_f1:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save(
                model.state_dict(),
                save_dir / 'av_cat_b_bce_only_best.pt'
            )
            print(f"  -> New best (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final eval with threshold optimization
    print(f"\nBest dev F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(
        save_dir / 'av_cat_b_bce_only_best.pt',
        weights_only=True
    ))
    preds, probs, true = evaluate(
        model, dev_loader, device
    )
    y_dev = np.array(dev_labels)

    best_tf1, best_t = 0, 0.5
    for t in np.arange(0.30, 0.70, 0.005):
        f1 = f1_score(
            y_dev, (probs > t).astype(int),
            average='macro'
        )
        if f1 > best_tf1:
            best_tf1 = f1
            best_t = t

    final_preds = (probs > best_t).astype(int)
    print(f"Threshold optimized: F1={best_tf1:.4f} t={best_t:.3f}")

    metrics = compute_all_metrics(y_dev, final_preds)
    print_metrics(metrics, "AV Cat B BCE-Only — Final")

    pred_path = (
        PROJECT_ROOT / 'predictions'
        / 'av_Group_34_B_bce_only.csv'
    )
    pred_path.parent.mkdir(exist_ok=True)
    save_predictions(final_preds, pred_path)

    np.save(
        PROJECT_ROOT / 'predictions'
        / 'av_cat_b_bce_only_probs.npy', probs
    )

    for n, bl in [('SVM', 0.5610), ('LSTM', 0.6226),
                  ('BERT', 0.7854)]:
        gap = metrics['macro_f1'] - bl
        s = "BEATS" if gap > 0 else "BELOW"
        print(f"  vs {n}: {s} by {gap:+.4f}")
    print(f"Current best AV Cat B: 0.7422")
    print("Done!")


if __name__ == '__main__':
    main()
