"""AV Cat B — Higher char perturbation rate (0.10 vs 0.05).

Tests whether more aggressive augmentation helps generalize.
Also uses LR=2e-4 (best from previous iteration).
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
    AVCharDataset, augment_text, char_encode,
    generate_topic_labels, VOCAB_SIZE
)
from src.data_utils import (
    load_av_data, load_solution_labels, save_predictions
)
from src.scorer import compute_all_metrics, print_metrics


class AVCharDatasetV2(AVCharDataset):
    """AVCharDataset with configurable perturbation rate."""
    def __init__(self, df, max_len=1500, augment=False,
                 topic_labels=None, perturb_prob=0.10):
        super().__init__(
            df, max_len=max_len, augment=augment,
            topic_labels=topic_labels
        )
        self.perturb_prob = perturb_prob

    def __getitem__(self, idx):
        ids_1 = self.encoded_1[idx]
        ids_2 = self.encoded_2[idx]
        if self.augment:
            ids_1 = augment_text(
                ids_1, perturb_prob=self.perturb_prob
            )
            ids_2 = augment_text(
                ids_2, perturb_prob=self.perturb_prob
            )
        item = {
            'char_ids_1': torch.tensor(
                ids_1, dtype=torch.long
            ),
            'char_ids_2': torch.tensor(
                ids_2, dtype=torch.long
            ),
            'label': torch.tensor(
                self.labels[idx], dtype=torch.float
            ),
        }
        if self.topic_labels is not None:
            item['topic'] = torch.tensor(
                self.topic_labels[idx], dtype=torch.long
            )
        return item


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
    print("  AV Cat B — Perturb 0.10 Training")
    print("=" * 60)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')

    # Generate topic labels for contrastive/GRL
    all_texts = (
        list(train_df['text_1']) + list(train_df['text_2'])
    )
    topic_all = generate_topic_labels(
        all_texts, n_clusters=10
    )
    train_topic = topic_all[:len(train_df)]
    num_topics = int(topic_all.max()) + 1

    train_dataset = AVCharDatasetV2(
        train_df, max_len=1500, augment=True,
        topic_labels=train_topic, perturb_prob=0.10
    )
    dev_dataset = AVCharDatasetV2(
        dev_df, max_len=1500, augment=False,
        topic_labels=None, perturb_prob=0.0
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
        proj_dim=128, num_topics=num_topics,
        grl_lambda=0.0,
    ).to(device)

