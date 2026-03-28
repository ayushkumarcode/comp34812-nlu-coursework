"""NLI Category A — Test inference with stacking ensemble (refit features)."""
import sys
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import clean_text, save_predictions
from src.nli_pipeline import NLIFeatureExtractor


def main():
    print("=" * 60)
    print("  NLI Cat A — Test Inference")
    print("=" * 60, flush=True)

    model_dir = PROJECT_ROOT / 'models'
    train_path = PROJECT_ROOT / 'training_extracted' / 'training_data' / 'NLI' / 'train.csv'
    test_path = PROJECT_ROOT / 'test_data_nli.csv'
    out_path = PROJECT_ROOT / 'predictions' / 'nli_test_A.csv'
    out_path.parent.mkdir(exist_ok=True)

    # Load training data (needed to fit TF-IDF)
    print("\n[1/6] Loading training data...", flush=True)
    t0 = time.time()
    train_df = pd.read_csv(train_path, quotechar='"', engine='python')
    train_df['premise'] = train_df['premise'].apply(lambda x: clean_text(x, lowercase=False))
    train_df['hypothesis'] = train_df['hypothesis'].apply(lambda x: clean_text(x, lowercase=False))
    train_df['premise'] = train_df['premise'].apply(lambda x: '.' if not x or x.isspace() else x)
