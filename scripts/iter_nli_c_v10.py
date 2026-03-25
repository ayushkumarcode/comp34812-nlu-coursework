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
