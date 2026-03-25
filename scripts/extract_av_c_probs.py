"""Extract probabilities from saved AV Cat C models.

Loads each saved model checkpoint, runs inference,
saves probability arrays for ensembling.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (
    load_av_data, load_solution_labels
)
from sklearn.metrics import f1_score


class AVCEDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=384):
        self.t1 = list(df['text_1'])
        self.t2 = list(df['text_2'])
        self.labels = df['label'].values.astype(np.float32)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tok(
            self.t1[idx], self.t2[idx],
            truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(
                self.labels[idx], dtype=torch.float
            ),
        }


class AVCE(nn.Module):
    def __init__(self, mn='microsoft/deberta-v3-base'):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(mn)
        hs = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(hs, 256),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.classifier(out.last_hidden_state[:, 0])


def extract_probs(model, dataloader, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(ids, mask).squeeze(-1)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


def main():
    from transformers import AutoTokenizer

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    MN = 'microsoft/deberta-v3-base'
    tok = AutoTokenizer.from_pretrained(
        MN, use_fast=False
    )

    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')
    y_dev = np.array(dev_labels)

    dev_ds = AVCEDataset(dev_df, tok, max_len=384)
    dev_ds.labels = np.array(
        dev_labels, dtype=np.float32
    )
    dl = DataLoader(
        dev_ds, batch_size=16, shuffle=False,
        num_workers=4
    )

    models_dir = PROJECT_ROOT / 'models'
    preds_dir = PROJECT_ROOT / 'predictions'

    # Models to extract (simple cross-encoder architecture)
    checkpoints = {
        'crossenc': 'av_cat_c_crossenc_best.pt',
        'lr1e5': 'av_cat_c_lr1e5_best.pt',
        'rdrop': 'av_cat_c_rdrop_best.pt',
        'rdrop_v2': 'av_cat_c_rdrop_v2_best.pt',
        'fgm': 'av_cat_c_fgm_best.pt',
    }

    all_probs = {}
    for name, ckpt in checkpoints.items():
        ckpt_path = models_dir / ckpt
        out_path = preds_dir / f'av_cat_c_{name}_probs.npy'
        if out_path.exists():
            print(f"Already exists: {out_path}")
            p = np.load(out_path)
            all_probs[name] = p
            continue
        if not ckpt_path.exists():
            print(f"Not found: {ckpt_path}")
            continue
        print(f"Loading {name} from {ckpt}")
        model = AVCE(mn=MN).to(device)
        try:
            state = torch.load(
                ckpt_path, map_location=device,
                weights_only=True
            )
            model.load_state_dict(state)
        except Exception as e:
            print(f"  Failed to load: {e}")
