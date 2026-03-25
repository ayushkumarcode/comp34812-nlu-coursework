"""AV Cat C — DeBERTa with R-Drop + AWP combined.

Combines three regularization techniques:
1. R-Drop: consistency regularization
2. AWP: adversarial weight perturbation
3. Label Smoothing (0.05)

AWP starts after epoch 3 for stability.
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

from src.data_utils import (
    load_av_data, load_solution_labels, save_predictions
)
from src.scorer import compute_all_metrics, print_metrics
