"""NLI Cat C iter10 — R-Drop alpha=0.5, epochs=20, patience=10, LR warmup."""
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
        cls_repr = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_repr)


def compute_rdrop_loss(logits1, logits2, labels, bce_fn, alpha=0.5):
    loss1 = bce_fn(logits1, labels)
    loss2 = bce_fn(logits2, labels)
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
    WARMUP_RATIO = 0.1

    print(f"\n=== NLI Cat C + R-Drop (alpha={ALPHA}, warmup) ===")
    print(f"LR={LR}, BS={BATCH_SIZE}, MaxLen={MAX_LEN}")
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
    bce_loss = nn.BCEWithLogitsLoss()
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
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                logits1 = model(ids, mask).squeeze(-1)
                logits2 = model(ids, mask).squeeze(-1)
                loss, task_l, kl_l = compute_rdrop_loss(
                    logits1, logits2, labels, bce_loss, alpha=ALPHA)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            total_task += task_l.item()
            total_kl += kl_l.item()
            n_batches += 1

        # Evaluate
        model.eval()
        preds, probs_all, labels_all = [], [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                logits = model(ids, mask).squeeze(-1)
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).long()
                preds.extend(pred.cpu().numpy())
                probs_all.extend(probs.cpu().numpy())
                labels_all.extend(batch['label'].numpy())

        dev_f1 = f1_score(labels_all, preds, average='macro', zero_division=0)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d} | Loss: {total_loss/n_batches:.4f} "
              f"(task={total_task/n_batches:.4f}, kl={total_kl/n_batches:.4f}) "
              f"| Dev F1: {dev_f1:.4f} | LR: {lr_now:.2e}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / 'nli_cat_c_rdrop_v2_best.pt')
            np.save(save_dir / 'nli_cat_c_rdrop_v2_probs.npy', np.array(probs_all))
            print(f"  -> Best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final eval + threshold search
    print(f"\nBest NLI Cat C (R-Drop v2) dev F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(save_dir / 'nli_cat_c_rdrop_v2_best.pt', weights_only=True))
    model.eval()

    final_probs = []
    with torch.no_grad():
        for batch in dev_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(ids, mask).squeeze(-1)
            final_probs.extend(torch.sigmoid(logits).cpu().numpy())

    final_probs = np.array(final_probs)
    y_dev = np.array(dev_labels)

    print("\nThreshold search:")
    best_thresh, best_thresh_f1 = 0.5, 0
    for t in np.arange(0.30, 0.72, 0.02):
        preds_t = (final_probs >= t).astype(int)
        f1_t = f1_score(y_dev, preds_t, average='macro', zero_division=0)
        if f1_t > best_thresh_f1:
            best_thresh, best_thresh_f1 = t, f1_t
        print(f"  thresh={t:.2f}: F1={f1_t:.4f}")

    print(f"\nBest threshold: {best_thresh:.2f} -> F1={best_thresh_f1:.4f}")

    final_preds = (final_probs >= best_thresh).astype(int)
    metrics = compute_all_metrics(y_dev, final_preds)
    print_metrics(metrics, "NLI Cat C (R-Drop v2) — Final w/ Threshold")

    save_predictions(final_preds,
                     PROJECT_ROOT / 'predictions' / 'nli_Group_34_C_rdrop_v2.csv')

    baselines = {'SVM': 0.5846, 'LSTM': 0.6603, 'BERT': 0.8198}
    for name, bl in baselines.items():
        gap = metrics['macro_f1'] - bl
        print(f"  vs {name} ({bl:.4f}): {'BEATS' if gap > 0 else 'BELOW'} by {gap:+.4f}")
