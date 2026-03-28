"""
cat A training — lightgbm with ~695 stylometric features for AV (group 34).

to convert: python scripts/convert_to_ipynb.py notebooks/training_cat_a.py
"""

# %% [markdown]
# # Cat A — Training
# ## LightGBM with stylometric features
#
# this notebook trains our cat A solution: a LightGBM gradient boosting
# classifier using ~695 stylometric features per text pair. we've got
# novel features in there too — syntactic complexity, writing rhythm,
# and information-theoretic stuff.

# %%
# !pip install scikit-learn lightgbm spacy numpy pandas tqdm joblib
# !python -m spacy download en_core_web_md

# %%
import sys
import time
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.av_pipeline import AVFeatureExtractor
from src.scorer import compute_all_metrics, print_metrics

# %% [markdown]
# ## 1. Load data

# %%
train_df = load_av_data(split='train')
dev_df = load_av_data(split='dev')
y_train = train_df['label'].values
y_dev = np.array(load_solution_labels(task='av'))

print(f"Train: {len(train_df)} pairs")
print(f"Dev: {len(dev_df)} pairs")
print(f"Train label dist: {np.bincount(y_train)}")

# %% [markdown]
# ## 2. Feature extraction
#
# We extract ~468 per-text features across 9 groups:
# - Lexical (30), Character (56), TF-IDF+SVD (100), Function words (150)
# - POS tags (45), Structural (15), Syntactic complexity (10, novel)
# - Writing rhythm (6, novel), Information-theoretic (5, novel)
#
# Then we compute diff-vectors |f(text1) - f(text2)| plus style-only
# diffs plus 14 pairwise features, giving us ~695 total per pair.

# %%
extractor = AVFeatureExtractor(use_spacy=True, n_svd_components=100)
extractor.fit(train_df)

X_train, feature_names = extractor.transform(train_df)
X_dev, _ = extractor.transform(dev_df)

print(f"Train features: {X_train.shape}")
print(f"Dev features: {X_dev.shape}")
print(f"Feature count: {len(feature_names)}")

# %% [markdown]
# ## 3. Train LightGBM

# %%
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_dv = scaler.transform(X_dev)

model = LGBMClassifier(
    n_estimators=1000, max_depth=7, learning_rate=0.05,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1,
    verbose=-1, random_state=42, n_jobs=1,
)
model.fit(X_tr, y_train)

# %% [markdown]
# ## 4. Evaluate

# %%
y_pred = model.predict(X_dv)
metrics = compute_all_metrics(y_dev, y_pred)
print_metrics(metrics, "Cat A — Dev Set (LightGBM)")

# check against baselines
baselines = {'SVM': 0.5610, 'LSTM': 0.6226, 'BERT': 0.7854}
f1 = metrics['macro_f1']
for name, baseline in baselines.items():
    gap = f1 - baseline
    print(f"vs {name} ({baseline:.4f}): {'BEATS' if gap > 0 else 'BELOW'} by {gap:+.4f}")

# %% [markdown]
# ## 5. Save model

# %%
save_dir = Path('models')
save_dir.mkdir(exist_ok=True)
joblib.dump(model, save_dir / 'av_cat_a_lgbm.joblib')
joblib.dump(scaler, save_dir / 'av_cat_a_scaler.joblib')
joblib.dump(feature_names, save_dir / 'av_cat_a_feature_names.joblib')
joblib.dump(extractor.tfidf, save_dir / 'av_cat_a_tfidf.joblib')
joblib.dump(extractor.cosine, save_dir / 'av_cat_a_cosine.joblib')
print("all model artifacts saved to models/")

# %% [markdown]
# ## 6. Generate predictions

# %%
save_predictions(y_pred, 'predictions/Group_34_A.csv')
print("saved to predictions/Group_34_A.csv")
