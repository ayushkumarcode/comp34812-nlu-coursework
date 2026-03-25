"""AV Cat B iter3 — Char-CNN+BiLSTM+GRL with LR=2e-4, no contrastive.

Changes from v2 (0.7414):
- LR=2e-4 (was 5e-4) — even slower
- Disable contrastive loss entirely (only BCE + topic adversarial)
- patience=20 (was 15), max_epochs=120
"""
import sys, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.av_cat_b_model import AVCatBModel
from src.models.av_cat_b_dataset import AVCharDataset, generate_topic_labels, VOCAB_SIZE
from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics

def train_epoch(model, dl, opt, dev, bce_fn, topic_fn, t_weight=0.02):
    model.train()
    total, all_p, all_l = 0, [], []
    for b in dl:
        c1, c2, labels = b['char_ids_1'].to(dev), b['char_ids_2'].to(dev), b['label'].to(dev)
        opt.zero_grad()
        logits, tl, _ = model(c1, c2, return_embeddings=True)
