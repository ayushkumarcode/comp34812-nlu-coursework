"""
COMP34812 — Solution 1 (Category A) Demo / Inference Notebook
Group 34 — NLI Track

Demonstrates how to load the trained model and make predictions.
"""

# %% [markdown]
# # Solution 1 (Category A) — Demo / Inference
# ## Feature-Rich Stacking Ensemble for Natural Language Inference

# %%
# !pip install scikit-learn xgboost lightgbm numpy pandas joblib spacy
# !python -m spacy download en_core_web_sm

# %%
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import clean_text, save_predictions
from src.nli_pipeline import NLIFeatureExtractor
from src.training.train_nli_ensemble import load_ensemble, predict, predict_proba

# %% [markdown]
# ## 1. Load Trained Model

# %%
scaler, ensemble, tfidf, feature_names = load_ensemble('models')
print(f"Model loaded successfully. Features: {len(feature_names)}")

# %% [markdown]
# ## 2. Prepare Input Data

# %%
from src.data_utils import load_nli_data
test_df = load_nli_data(split='dev')
print(f"Test data: {len(test_df)} pairs")

# %% [markdown]
# ## 3. Extract Features and Predict

# %%
extractor = NLIFeatureExtractor(use_spacy=True)
extractor.tfidf = tfidf
extractor._fitted = True
extractor._feature_names = feature_names

X_test, _ = extractor.transform(test_df)
print(f"Feature matrix: {X_test.shape}")

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
for i in range(min(5, len(test_df))):
    premise = test_df.iloc[i]['premise'][:100] + "..."
    hypothesis = test_df.iloc[i]['hypothesis'][:100]
    label = "Entailed" if predictions[i] == 1 else "Not Entailed"
    print(f"\nPair {i+1}:")
    print(f"  Premise:    {premise}")
    print(f"  Hypothesis: {hypothesis}")
    print(f"  Prediction: {label} (prob={probabilities[i]:.3f})")
