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

        if t_weight > 0 and 'topic' in b:
            tgt = b['topic'].to(device)
            loss = loss + t_weight * (topic_fn(tl1, tgt) + topic_fn(tl2, tgt)) / 2

        loss.backward()

        # FGM adversarial step
        fgm.attack()
        adv_logits, _, _ = model(c1, c2, return_embeddings=True)
        adv_loss = bce_fn(adv_logits.squeeze(-1), labels)
        adv_loss.backward()
        fgm.restore()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        total_loss += loss.item()
        total_kl += kl.item()
        total_adv += adv_loss.item()
        avg_logits = (l1 + l2) / 2
        all_p.extend((torch.sigmoid(avg_logits) > 0.5).long().cpu().numpy())
        all_l.extend(labels.cpu().numpy())

    n = len(dl)
    f1 = f1_score(all_l, all_p, average='macro', zero_division=0)
    return total_loss / n, total_kl / n, total_adv / n, f1


def evaluate(model, dl, device):
    """Standard eval — single forward pass."""
    model.eval()
    all_p, all_pr, all_l = [], [], []
    with torch.no_grad():
        for b in dl:
            c1, c2 = b['char_ids_1'].to(device), b['char_ids_2'].to(device)
            logits, _ = model(c1, c2)
            pr = torch.sigmoid(logits.squeeze(-1))
            all_p.extend((pr > 0.5).long().cpu().numpy())
            all_pr.extend(pr.cpu().numpy())
            all_l.extend(b['label'].numpy())
    return np.array(all_p), np.array(all_pr), np.array(all_l)

def main():
    RDROP_ALPHA, FGM_EPSILON, LR = 0.5, 0.5, 2e-4
    MAX_EPOCHS, PATIENCE, BATCH_SIZE = 120, 20, 64

    print("=" * 60)
    print(f"  AV Cat B v4 — R-Drop(a={RDROP_ALPHA}) + FGM(e={FGM_EPSILON}), lr={LR}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n[1/5] Loading data...")
    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')
    print(f"  Train: {len(train_df)}, Dev: {len(dev_df)}")

    print("\n[2/5] Generating topic labels...")
    all_texts = list(train_df['text_1']) + list(train_df['text_2'])
    topic_all = generate_topic_labels(all_texts, n_clusters=10)
    train_topic = topic_all[:len(train_df)]
    num_topics = int(topic_all.max()) + 1

    print("\n[3/5] Creating datasets...")
    train_ds = AVCharDataset(train_df, max_len=1500, augment=True,
                             topic_labels=train_topic)
    dev_ds = AVCharDataset(dev_df, max_len=1500, augment=False, topic_labels=None)
    dev_ds.labels = np.array(dev_labels, dtype=np.float32)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
    dev_dl = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    print("\n[4/5] Building model...")
    model = AVCatBModel(
        vocab_size=VOCAB_SIZE, char_emb_dim=32, cnn_filters=128,
        lstm_hidden=128, proj_dim=128, num_topics=num_topics,
        grl_lambda=0.0,
    ).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    bce_fn = nn.BCEWithLogitsLoss()
    topic_fn = nn.CrossEntropyLoss()
    fgm = FGM(model, epsilon=FGM_EPSILON, emb_name='char_emb')
    opt = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)


if __name__ == '__main__':
    main()
