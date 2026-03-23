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
