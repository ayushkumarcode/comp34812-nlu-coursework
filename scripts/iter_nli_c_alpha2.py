"""NLI Cat C iter12 — R-Drop alpha=2.0, warmup, epochs=20.

Current best: 0.9252 with alpha=0.5.
Try alpha=2.0 for stronger consistency regularization.
"""
import sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics

class D(Dataset):
    def __init__(self, df, tok, ml=128):
        self.p, self.h = list(df['premise']), list(df['hypothesis'])
        self.l = df['label'].values.astype(np.float32)
        self.tok, self.ml = tok, ml
    def __len__(self): return len(self.l)
    def __getitem__(self, i):
        e = self.tok(self.p[i], self.h[i], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        return {'ids': e['input_ids'].squeeze(0), 'mask': e['attention_mask'].squeeze(0), 'label': torch.tensor(self.l[i], dtype=torch.float)}

class M(nn.Module):
