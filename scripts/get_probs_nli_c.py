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

