"""NLI Cat C iter11 — R-Drop alpha=0.5 + label smoothing 0.05.

Current best: 0.9252 (R-Drop alpha=0.5, warmup, epochs=20).
Try adding label smoothing to further regularize.
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

from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics


class NLIDeBERTaDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.premises = list(df['premise'])
        self.hypotheses = list(df['hypothesis'])
        self.labels = df['label'].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.premises[idx], self.hypotheses[idx],
            truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float),
        }


class NLIDeBERTaCrossEncoder(nn.Module):
    def __init__(self, model_name='microsoft/deberta-v3-base'):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])


def compute_rdrop_loss_smooth(logits1, logits2, labels, alpha=0.5, smooth=0.05):
    """R-Drop loss with label smoothing for binary classification."""
    # Smooth labels: y' = y * (1 - smooth) + 0.5 * smooth
    labels_smooth = labels * (1 - smooth) + 0.5 * smooth
    loss1 = F.binary_cross_entropy_with_logits(logits1, labels_smooth)
    loss2 = F.binary_cross_entropy_with_logits(logits2, labels_smooth)
    task_loss = (loss1 + loss2) / 2

    p1 = torch.sigmoid(logits1)
    p2 = torch.sigmoid(logits2)
    dist1 = torch.stack([p1, 1 - p1], dim=-1).clamp(min=1e-7)
    dist2 = torch.stack([p2, 1 - p2], dim=-1).clamp(min=1e-7)
    kl = (F.kl_div(dist1.log(), dist2, reduction='batchmean') +
          F.kl_div(dist2.log(), dist1, reduction='batchmean')) / 2
    return task_loss + alpha * kl, task_loss, kl


def main():
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    MODEL_NAME = 'microsoft/deberta-v3-base'
    LR = 1e-5
    BATCH_SIZE = 16
    MAX_LEN = 128
    EPOCHS = 20
    PATIENCE = 10
    ALPHA = 0.5
    SMOOTH = 0.05
    WARMUP_RATIO = 0.1

    print(f"\n=== NLI Cat C + R-Drop + Label Smoothing ===")
    print(f"LR={LR}, BS={BATCH_SIZE}, MaxLen={MAX_LEN}")
    print(f"Alpha={ALPHA}, Smooth={SMOOTH}")
    print(f"Epochs={EPOCHS}, Patience={PATIENCE}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_df = load_nli_data(split='train')
    dev_df = load_nli_data(split='dev')
    dev_labels = load_solution_labels(task='nli')

    train_dataset = NLIDeBERTaDataset(train_df, tokenizer, max_len=MAX_LEN)
    dev_dataset = NLIDeBERTaDataset(dev_df, tokenizer, max_len=MAX_LEN)
    dev_dataset.labels = np.array(dev_labels, dtype=np.float32)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = NLIDeBERTaCrossEncoder(model_name=MODEL_NAME).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = GradScaler('cuda')

    best_f1 = 0
    patience_counter = 0
    save_dir = PROJECT_ROOT / 'models'
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_task, total_kl = 0, 0, 0
        n_batches = 0

        for batch in train_loader:
            ids = batch['input_ids'].to(device)
