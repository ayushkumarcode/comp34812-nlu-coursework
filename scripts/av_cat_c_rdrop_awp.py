"""AV Cat C — DeBERTa with R-Drop + AWP combined.

Combines three regularization techniques:
1. R-Drop: consistency regularization
2. AWP: adversarial weight perturbation
3. Label Smoothing (0.05)

AWP starts after epoch 3 for stability.
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (
    load_av_data, load_solution_labels, save_predictions
)
from src.scorer import compute_all_metrics, print_metrics


class AVCEDataset(Dataset):
    """AV cross-encoder dataset."""
    def __init__(self, df, tokenizer, max_len=384):
        self.t1 = list(df['text_1'])
        self.t2 = list(df['text_2'])
        self.labels = df['label'].values.astype(np.float32)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tok(
            self.t1[idx], self.t2[idx],
            truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(
                self.labels[idx], dtype=torch.float
            ),
        }


class AVCE(nn.Module):
    """DeBERTa cross-encoder for AV."""
    def __init__(self, mn='microsoft/deberta-v3-base'):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(mn)
        hs = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(hs, 256),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.classifier(out.last_hidden_state[:, 0])


class AWP:
    """Adversarial Weight Perturbation."""
    def __init__(self, model, adv_lr=1e-2, adv_eps=1e-2):
        self.model = model
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup_eps = {}

    def attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (param.requires_grad and
                    param.grad is not None and
                    'LayerNorm' not in name and
                    'bias' not in name):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data)
                if norm1 != 0 and not torch.isnan(norm1):
                    r = self.adv_lr * param.grad / (
                        norm1 + e
                    ) * (norm2 + e)
                    r = torch.clamp(
                        r, -self.adv_eps, self.adv_eps
                    )
                    param.data.add_(r)
                    self.backup_eps[name] = r

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup_eps:
                param.data.sub_(self.backup_eps[name])
        self.backup_eps = {}


def smooth_bce(logits, labels, smoothing=0.05):
    """BCE with label smoothing."""
    sl = labels * (1 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, sl)


def rdrop_loss(l1, l2, labels, alpha=1.0, smooth=0.05):
    """R-Drop + label smoothing."""
    loss1 = smooth_bce(l1, labels, smooth)
    loss2 = smooth_bce(l2, labels, smooth)
    task = (loss1 + loss2) / 2
    p1, p2 = torch.sigmoid(l1), torch.sigmoid(l2)
    d1 = torch.stack([p1, 1 - p1], -1).clamp(1e-7)
    d2 = torch.stack([p2, 1 - p2], -1).clamp(1e-7)
    kl = (
        F.kl_div(d1.log(), d2, reduction='batchmean')
        + F.kl_div(d2.log(), d1, reduction='batchmean')
    ) / 2
    return task + alpha * kl


def main():
    from transformers import AutoTokenizer

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    MN = 'microsoft/deberta-v3-base'
    LR, BS, ML = 2e-5, 8, 384
    EPOCHS, PAT = 25, 7
    ALPHA = 1.0
    AWP_START = 3

    print(f"\n=== AV Cat C R-Drop+AWP ===")
    print(f"LR={LR} BS={BS} ML={ML} alpha={ALPHA}")
    print(f"AWP starts ep {AWP_START}\n")

    tok = AutoTokenizer.from_pretrained(MN, use_fast=False)
    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')

    train_ds = AVCEDataset(train_df, tok, max_len=ML)
    dev_ds = AVCEDataset(dev_df, tok, max_len=ML)
    dev_ds.labels = np.array(dev_labels, dtype=np.float32)

    tl = DataLoader(
        train_ds, batch_size=BS, shuffle=True,
        num_workers=4
    )
    dl = DataLoader(
        dev_ds, batch_size=BS, shuffle=False,
        num_workers=4
    )

    model = AVCE(mn=MN).to(device)
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': LR},
        {'params': model.classifier.parameters(),
         'lr': 5e-4},
    ], weight_decay=0.01)
    scaler = GradScaler('cuda')
    awp = AWP(model, adv_lr=1e-2, adv_eps=1e-2)

    best_f1, pat = 0, 0
    sd = PROJECT_ROOT / 'models'
    sd.mkdir(exist_ok=True)

    for ep in range(1, EPOCHS + 1):
        model.train()
        tl_sum, nb = 0, 0
        for batch in tl:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()

            # R-Drop: two forward passes
            with autocast('cuda'):
                l1 = model(ids, mask).squeeze(-1)
                l2 = model(ids, mask).squeeze(-1)
                loss = rdrop_loss(l1, l2, labels, ALPHA)
            scaler.scale(loss).backward()

            # AWP after warmup
            if ep >= AWP_START:
                scaler.unscale_(optimizer)
                awp.attack_step()
                with autocast('cuda'):
                    la1 = model(ids, mask).squeeze(-1)
                    la2 = model(ids, mask).squeeze(-1)
                    loss_a = rdrop_loss(
                        la1, la2, labels, ALPHA
                    )
                scaler.scale(loss_a).backward()
                awp.restore()
            else:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            tl_sum += loss.item()
            nb += 1

        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for batch in dl:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                logits = model(ids, mask).squeeze(-1)
                p = (torch.sigmoid(logits) > 0.5).long()
                preds.extend(p.cpu().numpy())
                labs.extend(batch['label'].numpy())

        dev_f1 = f1_score(
            labs, preds, average='macro', zero_division=0
        )
        awp_s = "AWP" if ep >= AWP_START else "no-awp"
        print(
            f"Ep {ep:3d} | "
            f"Loss: {tl_sum/nb:.4f} ({awp_s}) | "
            f"Dev F1: {dev_f1:.4f}"
        )

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            pat = 0
            torch.save(
                model.state_dict(),
                sd / 'av_cat_c_rdrop_awp_best.pt'
            )
            print(f"  -> Best (F1={best_f1:.4f})")
        else:
            pat += 1
            if pat >= PAT:
                print(f"Early stop at ep {ep}")
                break

    # Final eval + threshold
    print(f"\nBest F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(
        sd / 'av_cat_c_rdrop_awp_best.pt',
        weights_only=True
    ))
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dl:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(ids, mask).squeeze(-1)
            all_probs.extend(
                torch.sigmoid(logits).cpu().numpy()
            )
    all_probs = np.array(all_probs)
    y_dev = np.array(dev_labels)

    bf, bt = 0, 0.5
    for t in np.arange(0.30, 0.70, 0.005):
        f1 = f1_score(
            y_dev, (all_probs > t).astype(int),
            average='macro'
