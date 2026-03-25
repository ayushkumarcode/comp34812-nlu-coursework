"""AV Cat C — DeBERTa cross-encoder with AWP.

AWP (Adversarial Weight Perturbation, Wu et al. 2020):
Unlike FGM which perturbs embeddings, AWP perturbs the
model WEIGHTS in the direction that maximizes loss, then
trains on the perturbed model. More stable than FGM.
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

from src.data_utils import (
    load_av_data, load_solution_labels, save_predictions
)
from src.scorer import compute_all_metrics, print_metrics


class AVCEDataset(Dataset):
