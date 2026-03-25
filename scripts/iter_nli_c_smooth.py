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
