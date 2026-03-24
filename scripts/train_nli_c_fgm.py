"""NLI Cat C — DeBERTa Cross-Encoder with FGM Adversarial Training.

FGM (Fast Gradient Method, Miyato et al., 2017): Adds adversarial perturbation
to word embeddings during training. After the initial backward pass, compute
a perturbation r_adv = epsilon * grad / ||grad|| on the embedding layer,
then do a second forward+backward with the perturbed embeddings.

This improves model robustness and often boosts generalization.
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

from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics


class NLIDeBERTaDataset(Dataset):
