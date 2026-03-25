"""Get probability outputs for AV Cat C (cross-encoder) for threshold optimization.

Loads the best AV Cat C cross-encoder checkpoint and outputs probabilities
instead of binary predictions, then sweeps thresholds to find optimal.

Architecture: separate encoder + classifier saved as dict with keys
'encoder_state_dict' and 'classifier_state_dict'.
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

from src.data_utils import load_av_data, load_solution_labels, save_predictions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ---- Dataset class (must match train_av_cat_c_crossenc.py exactly) ----
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


def main():
    MODEL_NAME = 'microsoft/deberta-v3-base'
    MAX_LEN = 384
    BATCH_SIZE = 8

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    # Build model (must match train_av_cat_c_crossenc.py architecture exactly)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
    classifier = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(encoder.config.hidden_size, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
    ).to(device)

    # Load checkpoint (dict with encoder_state_dict and classifier_state_dict)
    checkpoint_path = PROJECT_ROOT / 'models' / 'av_cat_c_crossenc_best.pt'
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    encoder.eval()
    classifier.eval()
    print("Model loaded successfully.")

    # Load dev data
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')
    y_true = np.array(dev_labels)

    dev_dataset = AVCrossEncoderDataset(dev_df, tokenizer, max_len=MAX_LEN)
    dev_dataset.labels = np.array(dev_labels, dtype=np.float32)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Run inference and collect probabilities
    all_probs = []
    with torch.no_grad():
        for batch in dev_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            outputs = encoder(input_ids=ids, attention_mask=mask)
            cls_repr = outputs.last_hidden_state[:, 0, :]
            logits = classifier(cls_repr)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.extend(probs.cpu().numpy())

    probs = np.array(all_probs)
    print(f"Collected {len(probs)} probabilities.")
    print(f"Prob stats: min={probs.min():.4f}, max={probs.max():.4f}, "
          f"mean={probs.mean():.4f}, std={probs.std():.4f}")

    # Save probabilities
    prob_path = PROJECT_ROOT / 'predictions' / 'av_cat_c_probs.npy'
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
    pred_path = PROJECT_ROOT / 'predictions' / 'av_Group_34_C_thresh.csv'
    save_predictions(preds, pred_path)
    print(f"Optimized predictions saved to {pred_path}")

    # Detailed sweep around the optimum
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
