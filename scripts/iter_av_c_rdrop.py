"""AV Cat C iter6 — Cross-Encoder DeBERTa with R-Drop + LR=3e-5.

Changes from best (crossenc, LR=2e-5, no R-Drop):
- Add R-Drop regularization (alpha=1.0)
- LR=3e-5 for encoder (slightly faster learning)
- Epochs=20, patience=8
- Threshold search at end
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

from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics


class AVCrossEncoderDataset(Dataset):
