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

print("\nExtracting features (this takes a while)...")
t0 = time.time()
ext = AVFeatureExtractor(
    use_spacy=True, n_svd_components=100
)
ext.fit(train_df)
X_train, fnames = ext.transform(train_df)
X_dev, _ = ext.transform(dev_df)
print(f"Features extracted in {time.time()-t0:.1f}s")
print(f"Train: {X_train.shape}, Dev: {X_dev.shape}")

cache_dir = PROJECT_ROOT / 'models'
cache_dir.mkdir(exist_ok=True)

np.save(cache_dir / 'av_features_train.npy', X_train)
np.save(cache_dir / 'av_features_dev.npy', X_dev)
np.save(cache_dir / 'av_labels_train.npy', y_train)
np.save(cache_dir / 'av_labels_dev.npy', y_dev)

# Save feature names
with open(cache_dir / 'av_feature_names.txt', 'w') as f:
    for name in fnames:
        f.write(f"{name}\n")

print(f"\nCached to {cache_dir}")
print(f"  av_features_train.npy: {X_train.shape}")
print(f"  av_features_dev.npy: {X_dev.shape}")
print("Done!")
