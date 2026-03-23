# COMP34812 NLU Coursework — Group 34

## Natural Language Inference Shared Task

This repository contains our solutions for the COMP34812 NLU shared task on Natural Language Inference (NLI). Given a premise and hypothesis, the task is to determine whether the hypothesis is entailed (1) or not (0).

We submit two solutions from **two different categories** (A and C):

### Solution 1: Category A — Feature-Rich Stacking Ensemble

~280 features per premise-hypothesis pair including alignment and natural logic:
- Lexical overlap, semantic similarity, negation detection
- Word alignment (Sultan et al. 2014), natural logic (MacCartney & Manning 2007)
- XGBoost + LightGBM + SVM-RBF + LR → LR meta-learner

### Solution 2: Category C — DeBERTa-v3 Cross-Encoder (F1=0.9167)

Fine-tuned DeBERTa-v3-base as cross-encoder:
- Hypothesis-only adversarial debiasing via GRL
- Mixed precision training, early stopping
- Beats BERT baseline by +0.097 (p < 0.001)

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

- Training data: COMP34812 NLI shared task (provided by course)
- Pre-trained models: DeBERTa-v3-base (Microsoft), GloVe 6B 300d, spaCy en_core_web_sm
- No external datasets used (closed mode)

## Code Attribution

- scikit-learn (Pedregosa et al., 2011) — ML framework
- XGBoost (Chen & Guestrin, 2016), LightGBM (Ke et al., 2017) — gradient boosting
- spaCy (Honnibal & Montani, 2017) — NLP pipeline
- PyTorch (Paszke et al., 2019) — deep learning framework
- HuggingFace Transformers (Wolf et al., 2020) — DeBERTa model loading
- Gradient Reversal Layer adapted from Ganin & Lempitsky (2015)
- GloVe embeddings (Pennington et al., 2014) — pre-trained word vectors
- All code written by Group 34 unless otherwise noted
