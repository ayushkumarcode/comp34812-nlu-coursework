---

# Model Card for NLI-DeBERTa-CrossEncoder (Solution 2, Category C)

A fine-tuned DeBERTa-v3-base cross-encoder for binary Natural Language Inference, with hypothesis-only adversarial debiasing via Gradient Reversal Layer.


## Model Details

### Model Description

This model performs binary NLI: given a premise and hypothesis, it predicts whether the hypothesis is entailed (1) or not entailed (0). It uses DeBERTa-v3-base as a cross-encoder, processing the concatenated [CLS] premise [SEP] hypothesis [SEP] sequence. A secondary hypothesis-only encoder with Gradient Reversal Layer (GRL) provides adversarial debiasing to prevent hypothesis-only shortcut learning.

- **Developed by:** Group 34
- **Language(s):** English
- **Model type:** Fine-tuned transformer (cross-encoder)
- **Model architecture:** DeBERTa-v3-base cross-encoder + GRL hypothesis-only adversarial head
- **Finetuned from model:** microsoft/deberta-v3-base (86M parameters)

### Model Resources

- **Repository:** https://github.com/ayushkumarcode/comp34812-nlu-coursework
- **Paper or documentation:** He et al. (2021) "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"

## Training Details

### Training Data

COMP34812 NLI shared task training set: 24,432 premise-hypothesis pairs with binary labels (0=not entailed, 1=entailed). Near-balanced distribution (48.2% label 0, 51.8% label 1). Closed mode — no external datasets used.

### Training Procedure

#### Training Hyperparameters

- **Optimizer:** AdamW (lr=2e-5, weight_decay=0.01)
- **Batch size:** 32
- **Max sequence length:** 128 tokens (premise + hypothesis)
- **Max hypothesis length:** 48 tokens (for adversarial head)
- **Epochs:** 25 (early stopping, patience=5)
- **Best epoch:** 5
- **Loss:** BCEWithLogitsLoss + 0.1 * adversarial BCE
- **Mixed precision:** Yes (torch.amp)
- **Gradient clipping:** max_norm=1.0
- **GRL lambda:** 0.1

#### Speeds, Sizes, Times

- **Training time:** ~10 minutes on NVIDIA A2 (15GB)
- **Model size:** ~370MB (includes adversarial encoder)
- **Inference speed:** ~200 samples/second on GPU

## Evaluation

### Testing Data & Metrics

#### Testing Data

COMP34812 NLI dev set: 6,736 premise-hypothesis pairs.

#### Metrics

Primary: macro_f1. Secondary: accuracy, MCC.

### Results

| Metric | Value |
|--------|-------|
| accuracy | 0.9169 |
| macro_precision | 0.9177 |
| macro_recall | 0.9162 |
| **macro_f1** | **0.9167** |
| **MCC** | **0.8339** |

Beats SVM (+0.332), LSTM (+0.256), BERT (+0.097). All p < 0.001.
