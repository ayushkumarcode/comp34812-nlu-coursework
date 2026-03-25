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

