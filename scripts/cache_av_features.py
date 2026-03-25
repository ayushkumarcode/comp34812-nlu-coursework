"""Cache AV features to numpy files for faster loading.

Extracts all 695 features from train and dev sets,
saves as numpy arrays. Subsequent scripts can load
these directly instead of re-extracting.
"""
import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (
    load_av_data, load_solution_labels
)
from src.av_pipeline import AVFeatureExtractor

print("=== Caching AV Features ===\n")

t0 = time.time()
train_df = load_av_data(split='train')
dev_df = load_av_data(split='dev')
y_train = train_df['label'].values
y_dev = np.array(load_solution_labels(task='av'))
print(f"Data loaded in {time.time()-t0:.1f}s")

