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
