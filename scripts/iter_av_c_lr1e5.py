"""AV Cat C iter7 — Cross-Encoder DeBERTa LR=1e-5 (lower than current best 2e-5).

The current best AV Cat C (0.8074) used LR=2e-5, no R-Drop.
R-Drop with LR=3e-5 failed completely (model didn't learn).
Try LR=1e-5 with more epochs instead.
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
    def __init__(self, df, tokenizer, max_len=384):
        self.texts_1 = list(df['text_1'])
        self.texts_2 = list(df['text_2'])
