"""AV Category A — Test inference with pre-fitted LightGBM pipeline."""
import sys
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import clean_text, save_predictions
from src.av_pipeline import AVFeatureExtractor


def main():
    print("=" * 60)
    print("  AV Cat A — Test Inference")
    print("=" * 60, flush=True)

    model_dir = PROJECT_ROOT / 'models'
    test_path = PROJECT_ROOT / 'test_data_av.csv'
    out_path = PROJECT_ROOT / 'predictions' / 'av_test_A.csv'
    out_path.parent.mkdir(exist_ok=True)

    # Load test data
    print("\n[1/5] Loading test data...", flush=True)
    t0 = time.time()
