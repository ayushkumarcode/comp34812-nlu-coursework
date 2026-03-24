"""AV Cat C — Cross-Encoder DeBERTa with FGM Adversarial Training.

FGM (Fast Gradient Method, Miyato et al., 2017): Adds adversarial perturbation
to word embeddings during training. After the initial backward pass, compute
a perturbation r_adv = epsilon * grad / ||grad|| on the embedding layer,
then do a second forward+backward with the perturbed embeddings.

This improves model robustness and often boosts generalization.
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

from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics


class AVCrossEncoderDataset(Dataset):
    """AV dataset for cross-encoder: tokenize text_1 + text_2 as pair."""
    def __init__(self, df, tokenizer, max_len=384):
        self.texts_1 = list(df['text_1'])
        self.texts_2 = list(df['text_2'])
        self.labels = df['label'].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts_1[idx], self.texts_2[idx],
            truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float),
        }


class AVCrossEncoder(nn.Module):
    """Simple cross-encoder: DeBERTa + classification head."""
    def __init__(self, model_name='microsoft/deberta-v3-base'):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_repr)
        return logits


class FGM:
    """Fast Gradient Method for adversarial training on embeddings."""

    def __init__(self, model, epsilon=1.0, emb_name='word_embeddings'):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        """Add adversarial perturbation to word embedding weights."""
        for name, param in self.model.named_parameters():
            if self.emb_name in name and param.requires_grad and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = param.grad.norm()
                if norm != 0 and not torch.isnan(norm):
                    r_adv = self.epsilon * param.grad / norm
                    param.data.add_(r_adv)

    def restore(self):
        """Restore original embedding weights."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def main():
    from transformers import AutoTokenizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    MODEL_NAME = 'microsoft/deberta-v3-base'
    LR = 2e-5
    BATCH_SIZE = 8
    MAX_LEN = 384
    EPOCHS = 25
    PATIENCE = 7
    FGM_EPSILON = 1.0

    print(f"\n=== AV Cat C + FGM Adversarial Training (eps={FGM_EPSILON}) ===")
    print(f"LR={LR}, BS={BATCH_SIZE}, MaxLen={MAX_LEN}")
    print(f"Epochs={EPOCHS}, Patience={PATIENCE}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')

    train_dataset = AVCrossEncoderDataset(
        train_df, tokenizer, max_len=MAX_LEN)
    dev_dataset = AVCrossEncoderDataset(
        dev_df, tokenizer, max_len=MAX_LEN)
    dev_dataset.labels = np.array(dev_labels, dtype=np.float32)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = AVCrossEncoder(model_name=MODEL_NAME).to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': LR},
        {'params': model.classifier.parameters(), 'lr': 5e-4},
    ], weight_decay=0.01)
    scaler = GradScaler('cuda')
    fgm = FGM(model, epsilon=FGM_EPSILON)

    best_f1 = 0
    patience_counter = 0
    save_dir = PROJECT_ROOT / 'models'
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_adv_loss = 0, 0
        n_batches = 0

        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            # Standard forward + backward
            with autocast('cuda'):
                logits = model(ids, mask).squeeze(-1)
                loss = bce_loss(logits, labels)

            scaler.scale(loss).backward()

            # FGM adversarial attack on embeddings
            scaler.unscale_(optimizer)
            fgm.attack()

            with autocast('cuda'):
                logits_adv = model(ids, mask).squeeze(-1)
                loss_adv = bce_loss(logits_adv, labels)

            scaler.scale(loss_adv).backward()
            fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_adv_loss += loss_adv.item()
            n_batches += 1

        # Evaluate
        model.eval()
        preds, labels_all = [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                logits = model(ids, mask).squeeze(-1)
                pred = (torch.sigmoid(logits) > 0.5).long()
                preds.extend(pred.cpu().numpy())
                labels_all.extend(batch['label'].numpy())

        dev_f1 = f1_score(labels_all, preds, average='macro', zero_division=0)
        print(f"Epoch {epoch:3d} | Loss: {total_loss/n_batches:.4f} "
              f"(adv={total_adv_loss/n_batches:.4f}) | Dev F1: {dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(),
                       save_dir / 'av_cat_c_fgm_best.pt')
            print(f"  -> Best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation with best model
    print(f"\nBest AV Cat C (FGM) dev F1: {best_f1:.4f}")
    model.load_state_dict(torch.load(
        save_dir / 'av_cat_c_fgm_best.pt', weights_only=True))
    model.eval()
    final_preds = []
    with torch.no_grad():
        for batch in dev_loader:
