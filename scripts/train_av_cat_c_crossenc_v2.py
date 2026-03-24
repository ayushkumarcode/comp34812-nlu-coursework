"""AV Cat C — Cross-Encoder DeBERTa v2 with ScalarMix + GRL topic debiasing.

Creativity elements:
  1. ScalarMix layer weighting (Peters et al. 2018, ELMo-style)
  2. Gradient Reversal Layer for topic debiasing (Ganin et al. 2016)
  3. TF-IDF + KMeans topic pseudo-labels
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
from src.models.cat_c_deberta import ScalarMix, GRL
from src.models.av_cat_b_dataset import generate_topic_labels
