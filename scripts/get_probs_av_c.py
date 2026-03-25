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
