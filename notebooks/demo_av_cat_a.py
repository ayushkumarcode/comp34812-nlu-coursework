"""cat A demo — lightgbm with 736 stylometric features (group 34, av track)"""

# %% [markdown]
# # Cat A — Demo / Inference
# ## LightGBM with stylometric features
#
# this notebook shows how to run inference with our cat A model for
# authorship verification. it's a LightGBM classifier trained on ~736
# handcrafted stylometric features — stuff like syntactic complexity,
# writing rhythm, information-theoretic measures, FFT spectral analysis,
# zipf-mandelbrot law deviation, benford's law, and hurst exponents.

# %%
# !pip install scikit-learn lightgbm numpy pandas joblib spacy
# !python -m spacy download en_core_web_md

# %%
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_av_data, clean_text, save_predictions
from src.av_pipeline import AVFeatureExtractor

# set this to a csv path if you want to run on custom data, otherwise we use dev
INPUT_FILE = None  # e.g. 'test_data_av.csv'

# %% [markdown]
# ## 1. Load the trained model
#
# We're loading the pre-trained LightGBM, the scaler, and the feature
# name list. The model was trained on 27,643 text pairs with 736
# features per pair.

# %%
scaler = joblib.load('models/av_cat_a_scaler.joblib')
model = joblib.load('models/av_cat_a_lgbm.joblib')
feature_names = joblib.load('models/av_cat_a_feature_names.joblib')
print(f"Model loaded. Features: {len(feature_names)}")

# %% [markdown]
# ## 2. Load data and extract features
#
# We load the dev set (swap in test data path for final submission)
# and extract all 736 features using AVFeatureExtractor. This covers
# TF-IDF+SVD, spaCy POS/syntactic stuff, and our novel feature groups.

# %%
# grab data — use INPUT_FILE if set, otherwise fall back to dev split
if INPUT_FILE is not None:
    test_df = pd.read_csv(INPUT_FILE, quotechar='"', engine='python')
    test_df['text_1'] = test_df['text_1'].apply(
        lambda x: clean_text(x, lowercase=False))
    test_df['text_2'] = test_df['text_2'].apply(
        lambda x: clean_text(x, lowercase=False))
else:
    test_df = load_av_data(split='dev')
print(f"Test data: {len(test_df)} pairs")

extractor = AVFeatureExtractor(use_spacy=True, n_svd_components=100)
# load the pre-fitted tfidf stuff
extractor.tfidf = joblib.load('models/av_cat_a_tfidf.joblib')
extractor.cosine = joblib.load('models/av_cat_a_cosine.joblib')
extractor._fitted = True
extractor._feature_names = feature_names

X_test, _ = extractor.transform(test_df)
X_test_scaled = scaler.transform(X_test)
print(f"Features: {X_test_scaled.shape}")

# %% [markdown]
# ## 3. Generate predictions
#
# LightGBM predicts binary labels — 0 for different author,
# 1 for same author.

# %%
predictions = model.predict(X_test_scaled)
print(f"Predictions: {len(predictions)}")
print(f"Class distribution: {np.bincount(predictions)}")

# %% [markdown]
# ## 4. Save predictions

# %%
save_predictions(predictions, 'predictions/Group_34_A.csv')
print("Saved to predictions/Group_34_A.csv")
