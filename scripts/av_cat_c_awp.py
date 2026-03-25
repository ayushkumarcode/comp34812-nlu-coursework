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
