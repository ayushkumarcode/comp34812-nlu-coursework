"""Get probability outputs for AV Cat C (cross-encoder) for threshold optimization.

Loads the best AV Cat C cross-encoder checkpoint and outputs probabilities
instead of binary predictions, then sweeps thresholds to find optimal.

Architecture: separate encoder + classifier saved as dict with keys
'encoder_state_dict' and 'classifier_state_dict'.
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

from src.data_utils import load_av_data, load_solution_labels, save_predictions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ---- Dataset class (must match train_av_cat_c_crossenc.py exactly) ----
