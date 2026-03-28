"""AV Cat B — R-Drop + FGM adversarial training on Siamese char-CNN+BiLSTM+GRL.

Building on v3 (best F1 ~0.7123). Two improvements:
1. R-Drop (Liang et al., NeurIPS 2021): two forward passes with different
   dropout masks, KL divergence regularizes consistency.
2. FGM (Miyato et al., ICLR 2017): adversarial perturbation of char embeddings
   in the gradient direction to improve robustness.

Both techniques are cheap for this model since it's lightweight (~1M params).
"""
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.av_cat_b_model import AVCatBModel
from src.models.av_cat_b_dataset import AVCharDataset, generate_topic_labels, VOCAB_SIZE
from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics


if __name__ == '__main__':
    pass
