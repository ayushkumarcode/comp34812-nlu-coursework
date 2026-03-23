---

# Model Card for NLI-XGBoost-Features (Solution 1, Category A)

A feature-rich XGBoost classifier for binary NLI using ~213 handcrafted features.

## Model Details

### Model Description

This model performs binary NLI using handcrafted features extracted from premise-hypothesis pairs. Features span lexical overlap, semantic similarity (TF-IDF), negation/contradiction detection, syntactic structure, and interaction terms. A single XGBoost classifier (1000 trees) is trained on the scaled feature matrix.

- **Developed by:** Group 34
- **Language(s):** English
- **Model type:** Gradient boosted trees (XGBoost)
- **Model architecture:** Feature extraction pipeline + XGBoost classifier
- **Finetuned from model:** N/A (trained from scratch)

### Model Resources

- **Repository:** https://github.com/ayushkumarcode/comp34812-nlu-coursework
- **Paper or documentation:** Chen & Guestrin (2016) "XGBoost: A Scalable Tree Boosting System"

## Training Details

### Training Data

COMP34812 NLI shared task training set: 24,432 premise-hypothesis pairs. Near-balanced (48.2% label 0, 51.8% label 1). Closed mode.

### Training Procedure

#### Training Hyperparameters

- **n_estimators:** 1000
- **max_depth:** 7
- **learning_rate:** 0.05
- **subsample:** 0.8
- **colsample_bytree:** 0.8
- **Feature count:** 213
- **Feature scaling:** StandardScaler

#### Speeds, Sizes, Times

- **Feature extraction:** ~10 min
- **Training:** ~5 min on 1 CPU
- **Model size:** ~15MB

## Evaluation

### Testing Data & Metrics

COMP34812 NLI dev set: 6,736 pairs. Primary: macro_f1.

### Results

| Metric | Value |
|--------|-------|
| **macro_f1** | **0.6999** |
| accuracy | 0.7013 |
| **MCC** | **0.4016** |

Beats SVM (+0.115) and LSTM (+0.040).

## Technical Specifications

### Hardware

Training: 1 CPU core. Inference: any machine with Python.

### Software

Python 3.11, XGBoost, scikit-learn, spaCy en_core_web_sm.

## Bias, Risks, and Limitations

Closed-mode, English-only, surface features without deep semantics.

## Additional Information

Feature pipeline: src/nli_pipeline.py. Training: scripts/nli_cat_a_simple.py.

