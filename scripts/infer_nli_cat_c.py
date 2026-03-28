"""NLI Category C — Test inference with DeBERTa cross-encoder."""
import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import clean_text, save_predictions
from src.models.cat_c_deberta import NLIDeBERTaCrossEncoder


def main():
    print("=" * 60)
    print("  NLI Cat C — Test Inference")
    print("=" * 60, flush=True)

    model_path = PROJECT_ROOT / 'models' / 'nli_cat_c_best.pt'
    test_path = PROJECT_ROOT / 'test_data_nli.csv'
    out_path = PROJECT_ROOT / 'predictions' / 'nli_test_C.csv'
    out_path.parent.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}", flush=True)

    # Load test data
    print("\n[1/4] Loading test data...", flush=True)
    t0 = time.time()
    df = pd.read_csv(test_path, quotechar='"', engine='python')
    df['premise'] = df['premise'].apply(lambda x: clean_text(x, lowercase=False))
    df['hypothesis'] = df['hypothesis'].apply(lambda x: clean_text(x, lowercase=False))
    df['premise'] = df['premise'].apply(lambda x: '.' if not x or x.isspace() else x)
    print(f"  Loaded {len(df)} pairs in {time.time()-t0:.1f}s", flush=True)
