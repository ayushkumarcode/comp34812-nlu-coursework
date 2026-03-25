"""AV Cat B — Higher char perturbation rate (0.10 vs 0.05).

Tests whether more aggressive augmentation helps generalize.
Also uses LR=2e-4 (best from previous iteration).
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.av_cat_b_model import AVCatBModel
from src.models.av_cat_b_dataset import (
    AVCharDataset, augment_text, char_encode,
    generate_topic_labels, VOCAB_SIZE
)
from src.data_utils import (
    load_av_data, load_solution_labels, save_predictions
)
from src.scorer import compute_all_metrics, print_metrics


class AVCharDatasetV2(AVCharDataset):
    """AVCharDataset with configurable perturbation rate."""
    def __init__(self, df, max_len=1500, augment=False,
                 topic_labels=None, perturb_prob=0.10):
        super().__init__(
            df, max_len=max_len, augment=augment,
            topic_labels=topic_labels
        )
        self.perturb_prob = perturb_prob

    def __getitem__(self, idx):
        ids_1 = self.encoded_1[idx]
        ids_2 = self.encoded_2[idx]
        if self.augment:
            ids_1 = augment_text(
                ids_1, perturb_prob=self.perturb_prob
            )
            ids_2 = augment_text(
                ids_2, perturb_prob=self.perturb_prob
            )
        item = {
            'char_ids_1': torch.tensor(
                ids_1, dtype=torch.long
            ),
            'char_ids_2': torch.tensor(
                ids_2, dtype=torch.long
