# COMP34812 NLU Coursework — Group 34

## Authorship Verification Shared Task

This repository contains our solutions for the COMP34812 NLU shared task on Authorship Verification (AV). Given a pair of texts, the task is to determine whether they were written by the same author (1) or different authors (0).

We submit two solutions from **two different categories** (A and B):

### Solution 1: Category A — LightGBM with Comprehensive Stylometric Features (F1=0.7340)

~695 features per text pair across 9 feature groups:
- Lexical, character, function word, POS tag, structural features
- Novel: syntactic complexity profiling, writing rhythm analysis, information-theoretic signatures
- Diff-vector |f(text1) - f(text2)| + style-only diff-vector (topic-robustness)
- Pairwise: NCD (gzip/lzma/bz2), cosine similarity, JSD, Burrows' Delta
- LightGBM classifier (1000 trees, max_depth=7, lr=0.05)

### Solution 2: Category B — Adversarial Style-Content Disentanglement Network (F1=0.7422)

Siamese char-CNN + BiLSTM + GRL neural architecture:
- Character-level encoding (97-char vocab, max 1500 chars)
- Multi-width Conv1D (3,5,7 kernels, 128 filters each) + BiLSTM(128) + Additive Attention
- Gradient Reversal Layer for topic adversarial debiasing
- Stylistic invariance training (char perturbation + truncation augmentation)
- Beats LSTM baseline by +0.120 (statistically significant)

## Repository Structure

```
├── src/
│   ├── data_utils.py              # Data loading and preprocessing
│   ├── scorer.py                  # Scorer wrapper
│   ├── av_feature_engineering.py  # AV Cat A feature extraction (9 groups)
│   ├── av_tfidf_features.py       # TF-IDF + SVD features
│   ├── av_spacy_features.py       # spaCy-based POS + syntactic features
│   ├── av_pipeline.py             # Complete AV Cat A pipeline
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
- `models/av_cat_a_lgbm.joblib` (~1MB) — LightGBM classifier
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
