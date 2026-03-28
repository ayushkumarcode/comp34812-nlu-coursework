"""AV Category B — Test inference with Siamese char-CNN+BiLSTM+GRL."""
import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import clean_text, save_predictions
from src.models.av_cat_b_dataset import char_encode
from src.models.av_cat_b_model import AVCatBModel


def main():
    print("=" * 60)
    print("  AV Cat B — Test Inference")
    print("=" * 60, flush=True)

    model_path = PROJECT_ROOT / 'models' / 'av_cat_b_v3_best.pt'
    test_path = PROJECT_ROOT / 'test_data_av.csv'
    out_path = PROJECT_ROOT / 'predictions' / 'av_test_B.csv'
    out_path.parent.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}", flush=True)

    # Load test data
    print("\n[1/4] Loading test data...", flush=True)
    t0 = time.time()
    df = pd.read_csv(test_path, quotechar='"', engine='python')
    df['text_1'] = df['text_1'].apply(lambda x: clean_text(x, lowercase=False))
    df['text_2'] = df['text_2'].apply(lambda x: clean_text(x, lowercase=False))
    print(f"  Loaded {len(df)} pairs in {time.time()-t0:.1f}s", flush=True)

    # Encode characters
    print("\n[2/4] Encoding characters...", flush=True)
    t0 = time.time()
    max_len = 1500
    encoded_1 = [char_encode(t, max_len) for t in df['text_1']]
    encoded_2 = [char_encode(t, max_len) for t in df['text_2']]
    ids_1 = torch.tensor(np.array(encoded_1), dtype=torch.long)
    ids_2 = torch.tensor(np.array(encoded_2), dtype=torch.long)
    print(f"  Encoded in {time.time()-t0:.1f}s", flush=True)
