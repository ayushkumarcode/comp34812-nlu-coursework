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
    def __init__(self, mn='microsoft/deberta-v3-base'):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(mn)
        hs = self.enc.config.hidden_size
        self.cls = nn.Sequential(nn.Dropout(0.1), nn.Linear(hs, 256), nn.Tanh(), nn.Dropout(0.1), nn.Linear(256, 1))
    def forward(self, ids, mask):
        return self.cls(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :])

def rdrop(l1, l2, labels, alpha=2.0):
    bce = nn.BCEWithLogitsLoss()
    tl = (bce(l1, labels) + bce(l2, labels)) / 2
    p1, p2 = torch.sigmoid(l1), torch.sigmoid(l2)
    d1 = torch.stack([p1, 1-p1], -1).clamp(1e-7)
    d2 = torch.stack([p2, 1-p2], -1).clamp(1e-7)
    kl = (F.kl_div(d1.log(), d2, reduction='batchmean') + F.kl_div(d2.log(), d1, reduction='batchmean')) / 2
    return tl + alpha * kl, tl, kl

def main():
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")
    MN, LR, BS, ML, EP, PAT, ALPHA = 'microsoft/deberta-v3-base', 1e-5, 16, 128, 20, 10, 2.0
    print(f"\n=== NLI Cat C + R-Drop (alpha={ALPHA}) ===\nLR={LR}, BS={BS}, ML={ML}, EP={EP}\n")
    tok = AutoTokenizer.from_pretrained(MN, use_fast=False)
    tdf, ddf = load_nli_data(split='train'), load_nli_data(split='dev')
    dl = load_solution_labels(task='nli')
    tds, dds = D(tdf, tok, ML), D(ddf, tok, ML)
