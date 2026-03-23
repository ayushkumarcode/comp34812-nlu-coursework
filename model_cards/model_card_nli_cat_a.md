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
