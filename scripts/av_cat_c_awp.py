"""AV Cat C — DeBERTa cross-encoder with AWP.

AWP (Adversarial Weight Perturbation, Wu et al. 2020):
Unlike FGM which perturbs embeddings, AWP perturbs the
model WEIGHTS in the direction that maximizes loss, then
trains on the perturbed model. More stable than FGM.
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
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
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
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
    def __init__(self, model_name='microsoft/deberta-v3-base'):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
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
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


class AWP:
    """Adversarial Weight Perturbation.

    Perturbs model weights to maximize loss, then trains
    on the perturbed model. More stable than FGM.
    """
    def __init__(self, model, optimizer, adv_lr=1e-2,
                 adv_eps=1e-2):
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_step(self):
        """Perturb weights in gradient direction."""
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
                    # Clip perturbation
                    r = torch.clamp(
                        r, -self.adv_eps, self.adv_eps
                    )
                    param.data.add_(r)
                    self.backup[name] = param.data.clone()
                    self.backup_eps[name] = r

    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if name in self.backup_eps:
                param.data.sub_(self.backup_eps[name])
        self.backup = {}
        self.backup_eps = {}


def main():
    from transformers import AutoTokenizer

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    MODEL_NAME = 'microsoft/deberta-v3-base'
    LR, BS, MAX_LEN = 2e-5, 8, 384
    EPOCHS, PATIENCE = 25, 7
    AWP_LR, AWP_EPS = 1e-2, 1e-2
    AWP_START = 3  # Start AWP after epoch 3

    print(f"\n=== AV Cat C + AWP ===")
    print(f"LR={LR}, BS={BS}, MaxLen={MAX_LEN}")
    print(f"AWP_LR={AWP_LR}, AWP_EPS={AWP_EPS}")
    print(f"AWP starts at epoch {AWP_START}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=False
    )

    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')

    train_ds = AVCEDataset(
        train_df, tokenizer, max_len=MAX_LEN
    )
    dev_ds = AVCEDataset(
        dev_df, tokenizer, max_len=MAX_LEN
    )
    dev_ds.labels = np.array(dev_labels, dtype=np.float32)

    train_loader = DataLoader(
        train_ds, batch_size=BS, shuffle=True,
        num_workers=4
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=BS, shuffle=False,
        num_workers=4
    )

    model = AVCE(model_name=MODEL_NAME).to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': LR},
        {'params': model.classifier.parameters(),
         'lr': 5e-4},
    ], weight_decay=0.01)
    scaler = GradScaler('cuda')
    awp = AWP(model, optimizer, adv_lr=AWP_LR,
              adv_eps=AWP_EPS)

    best_f1, pat = 0, 0
    save_dir = PROJECT_ROOT / 'models'
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_adv = 0, 0
        n_b = 0

        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                logits = model(ids, mask).squeeze(-1)
                loss = bce(logits, labels)
            scaler.scale(loss).backward()
