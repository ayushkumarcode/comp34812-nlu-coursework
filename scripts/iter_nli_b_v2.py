"""NLI Cat B iter2 — ESIM + KIM with LR=2e-4, patience=10, hidden=300.

Changes from v1:
- LR=2e-4 (was 4e-4) — slower, more stable learning
- Patience=10 (was 7) — more time to converge
- max_epochs=60 (was 40) — more capacity
- Clip grad norm=5.0 (was 10.0) — more stable
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.nli_cat_b_model import ESIM
from src.models.nli_cat_b_dataset import (
    NLIVocabulary, NLIESIMDataset, load_glove_embeddings
)
from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics
