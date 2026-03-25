"""AV Cat C — R-Drop + Label Smoothing + Long Training.

The previous rdrop_smooth run was still improving at epoch
25, so train for 40 epochs with patience 10. Uses the
identical architecture and hyperparameters (alpha=0.5,
smoothing=0.05, LR=2e-5) that achieved 0.8247.
"""
import sys
import math
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


