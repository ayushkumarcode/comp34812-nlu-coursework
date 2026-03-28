"""AV Cat B — R-Drop + FGM adversarial training on Siamese char-CNN+BiLSTM+GRL.

Building on v3 (best F1 ~0.7123). Two improvements:
1. R-Drop (Liang et al., NeurIPS 2021): two forward passes with different
   dropout masks, KL divergence regularizes consistency.
2. FGM (Miyato et al., ICLR 2017): adversarial perturbation of char embeddings
   in the gradient direction to improve robustness.

Both techniques are cheap for this model since it's lightweight (~1M params).
"""
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
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


class FGM:
    """Fast Gradient Method — perturb char embeddings in gradient direction."""

    def __init__(self, model, epsilon=0.5, emb_name='char_emb'):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if self.emb_name in name and param.requires_grad and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = param.grad.norm()
                if norm != 0 and not torch.isnan(norm):
                    r_adv = self.epsilon * param.grad / norm
                    param.data.add_(r_adv)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def rdrop_kl(logits1, logits2):
    """Symmetric KL divergence for binary classification logits."""
    p1 = torch.sigmoid(logits1)
    p2 = torch.sigmoid(logits2)
    d1 = torch.stack([p1, 1 - p1], dim=-1).clamp(min=1e-7)
    d2 = torch.stack([p2, 1 - p2], dim=-1).clamp(min=1e-7)
    kl = (F.kl_div(d1.log(), d2, reduction='batchmean') +
          F.kl_div(d2.log(), d1, reduction='batchmean')) / 2
    return kl


def train_epoch(model, dl, opt, device, bce_fn, topic_fn, fgm,
                t_weight=0.02, rdrop_alpha=0.5):
    """One training epoch with R-Drop + FGM."""
    model.train()
    total_loss, total_kl, total_adv = 0, 0, 0
    all_p, all_l = [], []

    for b in dl:
        c1 = b['char_ids_1'].to(device)
        c2 = b['char_ids_2'].to(device)
        labels = b['label'].to(device)
        opt.zero_grad()

        # R-Drop: two forward passes with different dropout
        logits1, tl1, _ = model(c1, c2, return_embeddings=True)
        logits2, tl2, _ = model(c1, c2, return_embeddings=True)
        l1, l2 = logits1.squeeze(-1), logits2.squeeze(-1)

        # averaged BCE + KL consistency
        bce = (bce_fn(l1, labels) + bce_fn(l2, labels)) / 2
        kl = rdrop_kl(l1, l2)
        loss = bce + rdrop_alpha * kl

if __name__ == '__main__':
    pass
