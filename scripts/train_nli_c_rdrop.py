"""NLI Cat C — DeBERTa Cross-Encoder with R-Drop regularization.

R-Drop (Liang et al., 2021): Regularized Dropout for neural networks.
Two forward passes with different dropout masks, then minimize KL divergence
between their output distributions to improve consistency.

Total loss = (BCE1 + BCE2)/2 + alpha * symmetric_KL(p1, p2)
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
