---
{}
---

# Model Card for AV-StyleDisentangle: Adversarial Style-Content Disentanglement Network

A Siamese neural network for authorship verification that disentangles writing style from topic content using gradient reversal adversarial training, character-level CNN + BiLSTM encoding, additive attention, and contrastive embedding learning.

## Model Details

### Model Description

AV-StyleDisentangle is a Category B (neural network without pre-trained transformers) solution for the COMP34812 Authorship Verification shared task. The model processes each text as a character sequence through a shared (Siamese) encoder consisting of multi-width character-level CNNs, a BiLSTM, and additive attention, producing a fixed-size document embedding. Two text embeddings are compared via concatenation of [v1, v2, |v1-v2|, v1*v2] and classified through an MLP.

The central innovation is adversarial style-content disentanglement: a gradient reversal layer (GRL) on an auxiliary topic prediction head forces the encoder to produce representations that are NOT predictive of topic/domain, focusing purely on authorial style. This addresses the key confound in cross-domain AV where same-topic pairs are easily misclassified as same-author.

Additional contributions include:
1. **Contrastive embedding loss** (CosineEmbeddingLoss) as a primary training objective, shaping the embedding space so same-author pairs are close and different-author pairs are far
2. **Stylistic invariance training** via character perturbation augmentation and random truncation
3. **Interpretable attention weights** providing XAI explainability (Bahdanau attention)

- **Developed by:** Group 34
- **Language(s):** English
- **Model type:** Siamese neural network (Category B, no pre-trained transformers)
- **Model architecture:** Char Embedding(32) -> Multi-width Conv1D(3,5,7, 128 filters each) -> MaxPool -> BiLSTM(128) -> Additive Attention -> Projection(128) -> [v1,v2,|v1-v2|,v1*v2] -> MLP(512->256->64->1) + GRL Topic Head
- **Finetuned from model [optional]:** N/A (trained from scratch, no pre-trained embeddings)

### Model Resources

- **Repository:** https://github.com/ayushkumarcode/comp34812-nlu-coursework
- **Paper or documentation:**
  - Ganin & Lempitsky (2015) "Unsupervised Domain Adaptation by Backpropagation" — Gradient Reversal Layer
  - Boenninghoff et al. (2019, 2021) — O2D2 Siamese character-level AV system (PAN 2020/2021 winner)
  - Bromley et al. (1993) — Original Siamese network for signature verification
  - Gao et al. (2021) "SimCSE" — Contrastive learning for NLP
  - Bahdanau et al. (2015) — Additive attention mechanism
  - Zhang, Zhao & LeCun (2015) — Character-level CNNs
  - Kim (2014) — Multi-width CNN architecture

## Training Details

### Training Data

- **Dataset:** COMP34812 AV shared task training data
- **Size:** 27,643 text pairs
- **Label distribution:** ~50% same-author, ~50% different-author
- **Text sources:** Cross-domain (emails, blogs, movie reviews)
- **Preprocessing:** Character-level encoding (vocabulary ~97 chars: a-z, A-Z, 0-9, common punctuation, whitespace, UNK, PAD). No text normalization beyond URL replacement. Max sequence length: 1500 characters.
- **Topic pseudo-labels:** Generated via TF-IDF + K-Means clustering (10 clusters) or corpus-type heuristics (email/blog/review/unknown) for the adversarial head.

### Training Procedure

#### Architecture Details

| Component | Specification |
|-----------|--------------|
| Char Embedding | 97 chars, 32 dimensions |
| Conv1D (width 3) | 128 filters, padding=1 |
| Conv1D (width 5) | 128 filters, padding=2 |
| Conv1D (width 7) | 128 filters, padding=3 |
| MaxPool1d | kernel=3, stride=3 |
| BiLSTM | hidden=128, 1 layer, bidirectional |
| Attention | Additive (Bahdanau), query dim=128 |
| Projection | Linear(256->128), ReLU, Dropout(0.3) |
| MLP | 512->256(ReLU,Drop0.4)->64(ReLU,Drop0.3)->1 |
| Topic head | GRL(λ)->Linear(128->64,ReLU)->Linear(64->num_topics) |

**Total parameters:** ~2.5M

#### Training Hyperparameters

- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4
- **Scheduler:** CosineAnnealingWarmRestarts, T_0=5, T_mult=2, eta_min=1e-6
- **Loss:** BCE + 0.2 * CosineEmbeddingLoss(margin=0.3) + 0.1 * CrossEntropy(topic)
- **GRL schedule:** λ ramps linearly from 0 to 0.1 over epochs 1-5
- **Contrastive loss:** Introduced at epoch 2 (1 epoch of BCE-only warmup)
- **Batch size:** 64
- **Epochs:** Max 50, early stopping patience=7 on dev macro_f1
- **Gradient clipping:** max_norm=5.0
- **Augmentation:** Character perturbation (5% per-char), random truncation (80-100%)

#### Speeds, Sizes, Times

- **Training time:** ~1-3 hours on A100 GPU
- **Model size:** ~10MB (state_dict)
- **Inference speed:** ~500 pairs/second on GPU, ~50 pairs/second on CPU

## Evaluation

### Testing Data & Metrics

#### Testing Data

- **Dev set:** COMP34812 AV dev set (5,993 evaluation pairs)

#### Metrics

- **Primary:** macro_f1
- **Secondary:** accuracy, MCC, per-class precision/recall/F1
- **Statistical:** McNemar's test vs LSTM baseline, bootstrap 95% CIs

### Results

| Metric | Value |
|--------|-------|
| macro_f1 | [TO BE FILLED AFTER TRAINING] |
| accuracy | [TO BE FILLED] |
| MCC | [TO BE FILLED] |

**Baseline comparison:**

| Model | macro_f1 | vs Ours | McNemar p |
|-------|----------|---------|-----------|
| SVM | 0.5610 | [gap] | [p] |
| LSTM | 0.6226 | [gap] | [p] |
| BERT | 0.7854 | [gap] | [p] |

## Technical Specifications

### Hardware

- **Training:** CSF3 HPC cluster, NVIDIA A100 80GB GPU
- **Inference:** Any machine with Python 3.11+, PyTorch 2.0+, optional GPU

### Software

- Python 3.11
- PyTorch 2.5.1 (CUDA 12.1)
- scikit-learn (for topic label generation)
- numpy, pandas

## Bias, Risks, and Limitations

- **Character-level representation:** The model operates on raw characters without linguistic preprocessing. While this preserves stylistic signals (spacing, punctuation, casing), it may be sensitive to OCR artifacts or encoding inconsistencies.
- **Domain shift:** The GRL topic debiasing helps but may not generalize to entirely unseen domains (e.g., legal documents, code-switching texts).
- **Sequence length:** Texts are truncated to 1500 characters. Important stylistic features at the end of very long texts may be lost.
- **Contrastive loss collapse:** If not properly regularized, contrastive learning can cause embedding collapse (all embeddings becoming identical). We mitigate this with the margin parameter and delayed introduction.
- **Attention interpretability:** While attention weights provide some interpretability, they should not be interpreted as definitive feature importance (Jain & Wallace, 2019).

## Additional Information

- **Novel contributions:**
  1. Adversarial style-content disentanglement via GRL for AV (Ganin & Lempitsky 2015 adapted to AV domain)
  2. Contrastive embedding learning as primary (not auxiliary) training objective
  3. Stylistic invariance training via character perturbation augmentation
  4. Interpretable attention weights (XAI contribution via Bahdanau attention)

- **Code attribution:** PyTorch framework, GRL implementation adapted from Ganin & Lempitsky (2015) description. No external code copied directly.
