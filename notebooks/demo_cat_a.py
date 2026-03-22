"""
COMP34812 — Solution 1 (Category A) Demo / Inference Notebook
Group 34

Demonstrates how to load the trained model and make predictions.
"""

# %% [markdown]
# # Solution 1 (Category A) — Demo / Inference
# ## Loading the trained model and making predictions on new data

# %%
# !pip install scikit-learn xgboost numpy pandas joblib spacy
# !python -m spacy download en_core_web_md

# %%
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import clean_text, save_predictions
from src.av_pipeline import AVFeatureExtractor
from src.training.train_av_ensemble import load_ensemble, predict, predict_proba

# %% [markdown]
# ## 1. Load Trained Model

# %%
scaler, ensemble, tfidf, cosine, feature_names = load_ensemble('models')
print(f"Model loaded successfully. Features: {len(feature_names)}")

# %% [markdown]
# ## 2. Prepare Input Data

# %%
# Example: Load test data
# test_df = pd.read_csv('path/to/test.csv')
# For demo, we use dev data
from src.data_utils import load_av_data
test_df = load_av_data(split='dev')
print(f"Test data: {len(test_df)} pairs")

# %% [markdown]
# ## 3. Extract Features and Predict

# %%
# Reconstruct feature extractor with pre-fitted components
extractor = AVFeatureExtractor(use_spacy=True, n_svd_components=100)
extractor.tfidf = tfidf
extractor.cosine = cosine
extractor._fitted = True
extractor._feature_names = feature_names

# Extract features
X_test, _ = extractor.transform(test_df)
print(f"Feature matrix: {X_test.shape}")

# Make predictions
predictions = predict(X_test, scaler, ensemble)
probabilities = predict_proba(X_test, scaler, ensemble)

print(f"Predictions: {len(predictions)}")
print(f"Class distribution: {np.bincount(predictions)}")

# %% [markdown]
# ## 4. Save Predictions

# %%
save_predictions(predictions, 'predictions/Group_34_A.csv')
print("Predictions saved to predictions/Group_34_A.csv")

# %% [markdown]
# ## 5. Example Predictions

# %%
# Show some example predictions with probabilities
for i in range(min(5, len(test_df))):
    text1_preview = test_df.iloc[i]['text_1'][:80] + "..."
    text2_preview = test_df.iloc[i]['text_2'][:80] + "..."
    print(f"\nPair {i+1}:")
    print(f"  Text 1: {text1_preview}")
    print(f"  Text 2: {text2_preview}")
    print(f"  Prediction: {predictions[i]} (prob={probabilities[i]:.3f})")
