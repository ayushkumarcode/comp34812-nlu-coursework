"""cat A demo — lightgbm with 695 stylometric features (group 34, av track)"""

# %% [markdown]
# # Cat A — Demo / Inference
# ## LightGBM with stylometric features
#
# this notebook shows how to run inference with our cat A model for
# authorship verification. it's a LightGBM classifier trained on ~695
# handcrafted stylometric features — stuff like syntactic complexity,
# writing rhythm, and information-theoretic measures.

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

# Set INPUT_FILE to a CSV path for custom inference, or None for dev data.
INPUT_FILE = None  # e.g. 'test_data_av.csv'

# %% [markdown]
# ## 1. Load Trained Model
#
# We load the pre-trained LightGBM classifier, StandardScaler,
# and feature name list. The model was trained on 27,643 text pairs
# with 695 features per pair.

# %%
scaler = joblib.load('models/av_cat_a_scaler.joblib')
model = joblib.load('models/av_cat_a_lgbm.joblib')
feature_names = joblib.load('models/av_cat_a_feature_names.joblib')
print(f"Model loaded. Features: {len(feature_names)}")

# %% [markdown]
# ## 2. Load Test Data and Extract Features
#
# We load the dev set (replace with test data path for final submission)
# and extract all 695 stylometric features using the AVFeatureExtractor
# pipeline. This includes TF-IDF+SVD, spaCy POS/syntactic features,
# and all novel feature groups.

# %%
# Load data: use INPUT_FILE if set, otherwise default to dev split.
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
# Load pre-fitted TF-IDF components
extractor.tfidf = joblib.load('models/av_cat_a_tfidf.joblib')
extractor.cosine = joblib.load('models/av_cat_a_cosine.joblib')
extractor._fitted = True
extractor._feature_names = feature_names

X_test, _ = extractor.transform(test_df)
X_test_scaled = scaler.transform(X_test)
print(f"Features: {X_test_scaled.shape}")

# %% [markdown]
# ## 3. Generate Predictions
#
# The LightGBM model predicts binary labels (0 = different author,
# 1 = same author) for each text pair.

# %%
predictions = model.predict(X_test_scaled)
print(f"Predictions: {len(predictions)}")
print(f"Class distribution: {np.bincount(predictions)}")

# %% [markdown]
# ## 4. Save Predictions

# %%
save_predictions(predictions, 'predictions/Group_34_A.csv')
print("Saved to predictions/Group_34_A.csv")
