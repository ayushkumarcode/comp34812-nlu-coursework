# COMP34812 NLU Coursework — Group 34

## Natural Language Inference Shared Task

This repository contains our solutions for the COMP34812 NLU shared task on Natural Language Inference (NLI). Given a premise and hypothesis, the task is to determine whether the hypothesis is entailed (1) or not (0).

We submit two solutions from **two different categories** (A and C):

### Solution 1: Category A — Diff-Vector Stacking Ensemble (AV-StyleStack)

A traditional ML approach using ~950 stylometric features per text pair:
- 9 feature groups including novel syntactic complexity, writing rhythm, and information-theoretic features
- Diff-vector representation |f(text1) - f(text2)| with topic-robust style-only variant
- Stacking ensemble: SVM-RBF + Random Forest + XGBoost → Logistic Regression meta-learner

### Solution 2: Category B — Adversarial Style-Content Disentanglement Network (AV-StyleDisentangle)

A Siamese neural network with:
- Character-level multi-width CNN + BiLSTM + additive attention encoder
- Gradient Reversal Layer (GRL) for topic debiasing
- Contrastive embedding loss for style-space shaping
- Stylistic invariance training via character perturbation

## Repository Structure

```
├── src/
│   ├── data_utils.py              # Data loading and preprocessing
│   ├── scorer.py                  # Scorer wrapper
│   ├── av_feature_engineering.py  # AV Cat A feature extraction
│   ├── av_tfidf_features.py       # TF-IDF + SVD features
│   ├── av_spacy_features.py       # spaCy-based POS + syntactic features
│   ├── av_pipeline.py             # Complete AV Cat A pipeline
│   ├── nli_feature_engineering.py # NLI Cat A features
│   ├── nli_spacy_features.py      # NLI spaCy features
│   ├── nli_tfidf_features.py      # NLI TF-IDF features
│   ├── nli_pipeline.py            # NLI Cat A pipeline
│   ├── models/
│   │   ├── av_cat_b_model.py      # AV Cat B neural model
│   │   ├── av_cat_b_dataset.py    # AV Cat B dataset
│   │   ├── nli_cat_b_model.py     # NLI Cat B ESIM model
│   │   ├── nli_cat_b_dataset.py   # NLI Cat B dataset
│   │   └── cat_c_deberta.py       # Cat C DeBERTa models
│   ├── training/
│   │   ├── train_av_ensemble.py   # AV Cat A ensemble training
│   │   ├── train_nli_ensemble.py  # NLI Cat A ensemble training
│   │   ├── train_av_cat_b.py      # AV Cat B neural training
│   │   ├── train_nli_cat_b.py     # NLI Cat B ESIM training
│   │   └── train_cat_c.py         # Cat C DeBERTa training
│   └── evaluation/
│       └── eval_utils.py          # Evaluation utilities
├── notebooks/
│   ├── training_cat_a.py          # Cat A training notebook
│   ├── training_cat_b.py          # Cat B training notebook
│   ├── demo_cat_a.py              # Cat A inference demo
│   ├── demo_cat_b.py              # Cat B inference demo
│   └── evaluation.py              # Comprehensive evaluation
├── model_cards/
│   ├── model_card_cat_a.md        # Model card for Solution 1
│   └── model_card_cat_b.md        # Model card for Solution 2
├── predictions/
│   ├── Group_34_A.csv             # Solution 1 predictions
│   └── Group_34_B.csv             # Solution 2 predictions
├── scripts/                       # Slurm batch scripts for CSF3
└── models/                        # Trained model files (>10MB on OneDrive)
```

## Reproduction

### Requirements

```
pip install torch scikit-learn xgboost lightgbm spacy numpy pandas tqdm joblib transformers
python -m spacy download en_core_web_md
```

### Training

```bash
# Solution 1 (Category A) — CPU
python -m src.training.run_av_cat_a

# Solution 2 (Category B) — GPU required
python -m src.training.train_av_cat_b
```

### Inference

See `notebooks/demo_cat_a.py` and `notebooks/demo_cat_b.py` for demo code.

## Trained Models

Models larger than 10MB are stored on OneDrive:
- [Link to be added after training]

## Data Sources

- Training data: COMP34812 AV shared task (provided by course)
- Pre-trained models: spaCy en_core_web_md (general-purpose English NLP model)
- No external datasets used (closed mode)

## Code Attribution

- scikit-learn (Pedregosa et al., 2011) — ML framework
- XGBoost (Chen & Guestrin, 2016) — gradient boosting
- spaCy (Honnibal & Montani, 2017) — NLP pipeline
- PyTorch (Paszke et al., 2019) — deep learning framework
- Gradient Reversal Layer implementation adapted from Ganin & Lempitsky (2015)
- All code written by Group 34 unless otherwise noted
