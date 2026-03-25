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
        loss = bce_fn(logits.squeeze(-1), labels)
        if t_weight > 0 and 'topic' in b:
            loss = loss + t_weight * topic_fn(tl, b['topic'].to(dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item()
        all_p.extend((torch.sigmoid(logits.squeeze(-1)) > 0.5).long().cpu().numpy())
        all_l.extend(labels.cpu().numpy())
    return total / len(dl), f1_score(all_l, all_p, average='macro', zero_division=0)

def evaluate(model, dl, dev):
    model.eval()
    all_p, all_pr, all_l = [], [], []
    with torch.no_grad():
        for b in dl:
            c1, c2 = b['char_ids_1'].to(dev), b['char_ids_2'].to(dev)
            logits, _ = model(c1, c2)
            pr = torch.sigmoid(logits.squeeze(-1))
            all_p.extend((pr > 0.5).long().cpu().numpy())
            all_pr.extend(pr.cpu().numpy())
            all_l.extend(b['label'].numpy())
    return np.array(all_p), np.array(all_pr), np.array(all_l)

def main():
    print("=" * 60)
    print("  AV Cat B v3 — LR=2e-4, no contrastive")
    print("=" * 60)
