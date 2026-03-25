"""Get probability outputs for NLI Cat C (R-Drop model) for threshold optimization.

Loads the best NLI Cat C R-Drop checkpoint and outputs probabilities
instead of binary predictions, then sweeps thresholds to find optimal.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel

from src.data_utils import load_nli_data, load_solution_labels, save_predictions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ---- Model class (must match train_nli_c_rdrop.py exactly) ----
class NLIDeBERTaCrossEncoder(nn.Module):
    """Cross-encoder DeBERTa for NLI (no adversarial head -- clean R-Drop version)."""

    def __init__(self, model_name='microsoft/deberta-v3-base'):
        super().__init__()
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
        logits = self.classifier(cls_repr)
        return logits


# ---- Dataset class (must match train_nli_c_rdrop.py exactly) ----
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


def main():
    MODEL_NAME = 'microsoft/deberta-v3-base'
    MAX_LEN = 128
    BATCH_SIZE = 32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    # Load model
    model = NLIDeBERTaCrossEncoder(model_name=MODEL_NAME).to(device)
    checkpoint_path = PROJECT_ROOT / 'models' / 'nli_cat_c_rdrop_best.pt'
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully.")

    # Load dev data
    dev_df = load_nli_data(split='dev')
    dev_labels = load_solution_labels(task='nli')
    y_true = np.array(dev_labels)

    dev_dataset = NLIDeBERTaDataset(dev_df, tokenizer, max_len=MAX_LEN)
    dev_dataset.labels = np.array(dev_labels, dtype=np.float32)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Run inference and collect probabilities
    all_probs = []
    with torch.no_grad():
        for batch in dev_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(ids, mask).squeeze(-1)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())

    probs = np.array(all_probs)
    print(f"Collected {len(probs)} probabilities.")
    print(f"Prob stats: min={probs.min():.4f}, max={probs.max():.4f}, "
          f"mean={probs.mean():.4f}, std={probs.std():.4f}")

    # Save probabilities
    prob_path = PROJECT_ROOT / 'predictions' / 'nli_cat_c_probs.npy'
    prob_path.parent.mkdir(exist_ok=True)
    np.save(str(prob_path), probs)
    print(f"Probabilities saved to {prob_path}")

    # Sweep thresholds
    print("\n=== Threshold Sweep ===")
    best_f1 = 0
    best_thresh = 0.5
    default_f1 = None

    for thresh in np.arange(0.30, 0.71, 0.01):
        preds = (probs > thresh).astype(int)
        f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        if abs(thresh - 0.5) < 0.005:
            default_f1 = f1
            print(f"  thresh=0.50: F1={f1:.4f}  <-- default")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\nBest threshold: {best_thresh:.2f} -> F1={best_f1:.4f}")
    if default_f1 is not None:
        print(f"Improvement over 0.50: {best_f1 - default_f1:+.4f}")

    # Save predictions with optimal threshold
    preds = (probs > best_thresh).astype(int)
    pred_path = PROJECT_ROOT / 'predictions' / 'nli_Group_34_C_thresh.csv'
    save_predictions(preds, pred_path)
    print(f"Optimized predictions saved to {pred_path}")

    # Also print per-threshold results around the optimum
    print("\n=== Detailed sweep around optimum ===")
    for thresh in np.arange(max(0.30, best_thresh - 0.05),
                             min(0.71, best_thresh + 0.06), 0.01):
        preds = (probs > thresh).astype(int)
        f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        marker = " <-- BEST" if abs(thresh - best_thresh) < 0.005 else ""
        print(f"  thresh={thresh:.2f}: F1={f1:.4f}{marker}")

    print("\nDone!")


if __name__ == '__main__':
    main()
