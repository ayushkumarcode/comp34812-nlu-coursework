"""AV Cat B iter2 — Char-CNN+BiLSTM+GRL with lower LR and longer warmup.

Changes from v1:
- LR=5e-4 (was 1e-3) — slower, more stable
- BCE warmup=20 epochs (was 15)
- Contrastive from epoch 25 (was 15)
- Topic adversarial from epoch 15 (was 10)
- patience=15 (was 12)
- max_epochs=100 (was 80)
- Threshold search at end
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
    AVCharDataset, generate_topic_labels, VOCAB_SIZE
