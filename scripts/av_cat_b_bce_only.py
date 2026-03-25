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
