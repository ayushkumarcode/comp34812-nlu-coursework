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
    df = pd.read_csv(test_path, quotechar='"', engine='python')
    df['text_1'] = df['text_1'].apply(lambda x: clean_text(x, lowercase=False))
    df['text_2'] = df['text_2'].apply(lambda x: clean_text(x, lowercase=False))
    print(f"  Loaded {len(df)} pairs in {time.time()-t0:.1f}s", flush=True)

    # Load pre-fitted components
    print("\n[2/5] Loading model artifacts...", flush=True)
    lgbm = joblib.load(model_dir / 'av_cat_a_lgbm.joblib')
    scaler = joblib.load(model_dir / 'av_cat_a_scaler.joblib')
    saved_names = joblib.load(model_dir / 'av_cat_a_feature_names.joblib')
    saved_tfidf = joblib.load(model_dir / 'av_cat_a_tfidf.joblib')
    saved_cosine = joblib.load(model_dir / 'av_cat_a_cosine.joblib')
    print(f"  Feature count: {len(saved_names)}", flush=True)

    # Extract features using pre-fitted TF-IDF and cosine
    print("\n[3/5] Extracting features...", flush=True)
    t0 = time.time()
