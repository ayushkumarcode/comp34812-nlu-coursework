# COMP34812 NLU Coursework -- Group 34

## Authorship Verification Shared Task

This repo has our two solutions for the COMP34812 AV shared task. The goal's pretty straightforward: given two texts, predict whether they're by the same author (1) or not (0).

We're submitting one solution from Category A and one from Category B:

### Solution 1: Cat A -- LightGBM + Stylometric Features (F1=0.7340)

We built about 695 handcrafted features per text pair from 9 groups:
- The usual stuff: lexical, character-level, function words, POS tags, structural
- Three novel groups we came up with: syntactic complexity profiling, writing rhythm, and information-theoretic signatures
- Diff-vector |f(text1) - f(text2)| plus a style-only diff-vector for topic robustness
- Pairwise similarity measures: NCD (gzip/lzma/bz2), cosine sim, JSD, Burrows' Delta
- LightGBM on top (1000 trees, max_depth=7, lr=0.05)

### Solution 2: Cat B -- Siamese Char-CNN + BiLSTM with Adversarial Debiasing (F1=0.7422)

This one's a neural approach, built from scratch:
- Character-level encoding with a 97-char vocab, up to 1500 chars
- Multi-width Conv1D (kernel sizes 3, 5, 7 with 128 filters each), then BiLSTM + additive attention
- Gradient Reversal Layer (Ganin & Lempitsky 2015) to debias away topic signal
- Data augmentation via random char perturbation and truncation
- Beats the LSTM baseline by +0.120 (statistically significant, McNemar's p < 0.01)

## Repo Layout

```
src/
  data_utils.py              -- loading + cleaning data
  scorer.py                  -- wrapper around official scorer metrics
  av_feature_engineering.py  -- all 9 feature groups for Cat A
  av_tfidf_features.py       -- char n-gram TF-IDF + SVD
  av_spacy_features.py       -- POS tags + syntactic features via spaCy
  av_pipeline.py             -- ties everything together for Cat A
│   ├── models/
│   │   ├── av_cat_b_model.py      # AV Cat B neural model
│   │   ├── av_cat_b_dataset.py    # AV Cat B dataset + char encoding
│   │   └── cat_c_deberta.py       # Cat C DeBERTa models
│   ├── training/
│   │   ├── train_av_ensemble.py   # AV Cat A ensemble training
│   │   └── train_av_cat_b.py      # AV Cat B neural training
│   └── evaluation/
│       └── eval_utils.py          # Evaluation utilities
├── notebooks/
│   ├── demo_av_cat_a.py/.ipynb    # Cat A inference demo
│   ├── demo_av_cat_b.py/.ipynb    # Cat B inference demo
│   ├── training_cat_a.py/.ipynb   # Cat A training notebook
│   ├── training_cat_b.py/.ipynb   # Cat B training notebook
│   └── evaluation.py/.ipynb       # Comprehensive evaluation
├── model_cards/
│   ├── model_card_sol1_cat_a.md   # Model card for Solution 1
│   └── model_card_sol2_cat_b.md   # Model card for Solution 2
├── predictions/
│   ├── Group_34_A.csv             # Solution 1 predictions
│   └── Group_34_B.csv             # Solution 2 predictions
├── scripts/                       # Slurm batch scripts for CSF3
└── models/                        # Trained model files
```

## Reproduction

### Requirements

```
pip install torch scikit-learn lightgbm spacy numpy pandas tqdm joblib
python -m spacy download en_core_web_md
```

### Training

```bash
# Solution 1 (Category A) — CPU only
python scripts/iter_av_a_lgbm.py

# Solution 2 (Category B) — GPU required
python scripts/iter_av_b_v3.py
```

### Inference

See `notebooks/demo_av_cat_a.py` and `notebooks/demo_av_cat_b.py`.

## Trained Models

Models stored on GitHub (this repository):
- `models/av_cat_b_best.pt` (3.1MB) — Siamese char-CNN+BiLSTM+GRL
- `models/av_cat_a_lgbm.joblib` (5.6MB) — LightGBM classifier
- `models/av_cat_a_scaler.joblib` — StandardScaler
- `models/av_cat_a_feature_names.joblib` — Feature name list
- `models/av_cat_a_tfidf.joblib` — Pre-fitted TF-IDF vectorizer
- `models/av_cat_a_cosine.joblib` — Pre-fitted cosine similarity features

## Data Sources

- Training data: COMP34812 AV shared task (provided by course)
- Pre-trained models: spaCy en_core_web_md (POS tagging + dependency parsing)
- No external datasets used (closed mode)

## Code Attribution

- scikit-learn (Pedregosa et al., 2011) — ML framework, StandardScaler
- LightGBM (Ke et al., 2017) — gradient boosting classifier
- spaCy (Honnibal & Montani, 2017) — NLP pipeline for POS/syntactic features
- PyTorch (Paszke et al., 2019) — deep learning framework
- Gradient Reversal Layer adapted from Ganin & Lempitsky (2015)
- All code written by Group 34 unless otherwise noted
