---

# Model Card for NLI-DeBERTa-CrossEncoder (Solution 2, Category C)

A fine-tuned DeBERTa-v3-base cross-encoder for binary Natural Language Inference, with hypothesis-only adversarial debiasing via Gradient Reversal Layer.


## Model Details

### Model Description

This model performs binary NLI: given a premise and hypothesis, it predicts whether the hypothesis is entailed (1) or not entailed (0). It uses DeBERTa-v3-base as a cross-encoder, processing the concatenated [CLS] premise [SEP] hypothesis [SEP] sequence. A secondary hypothesis-only encoder with Gradient Reversal Layer (GRL) provides adversarial debiasing to prevent hypothesis-only shortcut learning.

- **Developed by:** Group 34
- **Language(s):** English
- **Model type:** Fine-tuned transformer (cross-encoder)
