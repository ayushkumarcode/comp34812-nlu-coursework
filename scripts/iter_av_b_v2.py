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
