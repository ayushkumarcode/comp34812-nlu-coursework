---
{}
---

# Model Card for AV-StyleDisentangle: Adversarial Style-Content Disentanglement Network

A Siamese neural network for authorship verification that disentangles writing style from topic content using gradient reversal adversarial training, character-level CNN + BiLSTM encoding, and additive attention.

## Model Details

### Model Description

AV-StyleDisentangle is a Category B (neural network without pre-trained transformers) solution for the COMP34812 Authorship Verification shared task. The model processes each text as a character sequence through a shared (Siamese) encoder consisting of multi-width character-level CNNs (kernels 3, 5, 7), a BiLSTM, and additive (Bahdanau) attention, producing a fixed-size 128-dimensional document embedding. Two text embeddings are compared via concatenation of [v1, v2, |v1-v2|, v1*v2] (512 dimensions) and classified through an MLP.

The central innovation is adversarial style-content disentanglement: a gradient reversal layer (GRL) on an auxiliary topic prediction head forces the encoder to produce representations that are NOT predictive of topic/domain, focusing purely on authorial style. This addresses the key confound in cross-domain AV where same-topic pairs are easily misclassified as same-author.

Additional contributions include:
1. **Stylistic invariance training** via character perturbation augmentation (5% per-char) and random truncation (80-100%)
2. **Careful GRL scheduling** with linear lambda ramp from 0 to 0.05 over 20 epochs, and topic adversarial loss introduced only after epoch 15
3. **Interpretable attention weights** providing XAI explainability (Bahdanau attention)

- **Developed by:** Group 34
- **Language(s):** English
- **Model type:** Siamese neural network (Category B, no pre-trained transformers)
- **Model architecture:** Char Embedding(32) -> Multi-width Conv1D(3,5,7, 128 filters each) -> MaxPool -> BiLSTM(128, bidirectional) -> Additive Attention -> Projection(128) -> [v1,v2,|v1-v2|,v1*v2] -> MLP(512->256->64->1) + GRL Topic Head
- **Finetuned from model [optional]:** N/A (trained from scratch, no pre-trained embeddings)

### Model Resources

- **Repository:** https://github.com/ayushkumarcode/comp34812-nlu-coursework
- **Paper or documentation:**
  - Ganin & Lempitsky (2015) "Unsupervised Domain Adaptation by Backpropagation" -- Gradient Reversal Layer
  - Boenninghoff et al. (2019, 2021) -- O2D2 Siamese character-level AV system (PAN 2020/2021 winner)
  - Bromley et al. (1993) -- Original Siamese network for signature verification
  - Bahdanau et al. (2015) -- Additive attention mechanism
  - Zhang, Zhao & LeCun (2015) -- Character-level CNNs
  - Kim (2014) -- Multi-width CNN architecture

## Training Details

### Training Data

- **Dataset:** COMP34812 AV shared task training data
- **Size:** 27,643 text pairs
- **Label distribution:** ~50% same-author, ~50% different-author
- **Text sources:** Cross-domain (emails, blogs, movie reviews)
- **Preprocessing:** Character-level encoding (vocabulary size 97: a-z, A-Z, 0-9, common punctuation, whitespace, UNK, PAD). No text normalization beyond URL replacement. Max sequence length: 1500 characters.
- **Topic pseudo-labels:** Generated via TF-IDF (5000 features) + MiniBatch K-Means clustering (10 clusters) on all training texts, with heuristic corpus-type labels (email/blog/review/unknown) used when coverage is sufficient.

### Training Procedure

#### Architecture Details

| Component | Specification |
|-----------|--------------|
| Char Embedding | 97 chars, 32 dimensions, padding_idx=0 |
| Conv1D (width 3) | 128 filters, padding=1 |
| Conv1D (width 5) | 128 filters, padding=2 |
| Conv1D (width 7) | 128 filters, padding=3 |
| MaxPool1d | kernel=3, stride=3 |
| CNN Dropout | 0.2 |
| BiLSTM | hidden=128, 1 layer, bidirectional (output: 256) |
| Attention | Additive (Bahdanau), query dim=128 |
| Projection | Linear(256->128), ReLU, Dropout(0.3) |
| Comparison | [v1, v2, |v1-v2|, v1*v2] = 512 dimensions |
| MLP | 512->256(ReLU, Dropout 0.4)->64(ReLU, Dropout 0.3)->1 |
| Topic head | GRL(lambda)->Linear(128->64, ReLU, Dropout 0.3)->Linear(64->10) |

**Total parameters:** ~814K

#### Training Hyperparameters

- **Optimizer:** AdamW, lr=2e-4, weight_decay=1e-4
- **Scheduler:** CosineAnnealingWarmRestarts, T_0=30, T_mult=2
- **Loss:** BCEWithLogitsLoss + topic adversarial CrossEntropyLoss (NO contrastive loss)
- **Topic adversarial weight:** 0.02 (applied from epoch 15 onward)
- **GRL schedule:** lambda ramps linearly from 0 to 0.05 over epochs 1-20, then fixed at 0.05
- **Batch size:** 64
- **Epochs:** Max 120, early stopping patience=20 on dev macro_f1
- **Gradient clipping:** max_norm=5.0
- **Augmentation:** Character perturbation (5% per-char), random truncation (80-100%)
- **Max sequence length:** 1500 characters

#### Speeds, Sizes, Times

- **Training time:** ~1-3 hours on A100 GPU (120 max epochs)
- **Model size:** 3.3MB (state_dict)
- **Inference speed:** ~500 pairs/second on GPU, ~50 pairs/second on CPU

## Evaluation

### Testing Data & Metrics

#### Testing Data

- **Dev set:** COMP34812 AV dev set (5,993 evaluation pairs)
- **Class distribution:** ~50/50 same-author vs different-author

#### Metrics

- **Primary:** macro_f1 (macro-averaged F1 score)
- **Secondary:** accuracy, Matthews Correlation Coefficient (MCC)
- **Statistical:** McNemar's test vs baselines, bootstrap 95% confidence intervals

### Results

| Metric | Value |
|--------|-------|
| macro_f1 | 0.7422 |
| accuracy | 0.7424 |
| MCC | 0.4845 |

**Baseline comparison:**

| Model | macro_f1 | vs Ours (gap) |
|-------|----------|---------------|
| SVM baseline | 0.5610 | +0.1812 |
| LSTM baseline | 0.6226 | +0.1196 |
| BERT baseline | 0.7854 | -0.0432 |

The model significantly outperforms both the SVM and LSTM baselines. While it does not surpass the BERT transformer baseline, it achieves strong performance using only character-level features and no pre-trained language representations. The adversarial topic debiasing contributes to cross-domain robustness.

## Technical Specifications

### Hardware

- **Training:** CSF3 HPC cluster, NVIDIA A100 80GB GPU
- **Inference:** Any machine with Python 3.11+, PyTorch 2.0+, optional GPU

### Software

- Python 3.11
- PyTorch 2.5.1 (CUDA 12.1)
- scikit-learn (for topic label generation via TF-IDF + KMeans)
- numpy, pandas

## Bias, Risks, and Limitations

- **Character-level representation:** The model operates on raw characters without linguistic preprocessing. While this preserves stylistic signals (spacing, punctuation, casing), it may be sensitive to OCR artifacts or encoding inconsistencies.
- **Domain shift:** The GRL topic debiasing helps but may not generalize to entirely unseen domains (e.g., legal documents, code-switching texts).
- **Sequence length:** Texts are truncated to 1500 characters. Important stylistic features at the end of very long texts may be lost.
- **Attention interpretability:** While attention weights provide some interpretability, they should not be interpreted as definitive feature importance (Jain & Wallace, 2019).
- **Topic cluster quality:** Topic pseudo-labels from TF-IDF + KMeans may not perfectly capture domain boundaries, potentially limiting the adversarial debiasing effectiveness.

## Additional Information

- **Novel contributions:**
  1. Adversarial style-content disentanglement via GRL for AV (Ganin & Lempitsky 2015 adapted to AV domain)
  2. Careful GRL scheduling (slow lambda ramp over 20 epochs, delayed topic loss from epoch 15) to prevent training instability
  3. Stylistic invariance training via character perturbation augmentation
  4. Interpretable attention weights (XAI contribution via Bahdanau attention)

- **Design decisions:**
  - Contrastive loss was tested in earlier versions (v1, v2) but removed in the final model (v3) as BCE + topic adversarial loss alone yielded better performance
  - Learning rate was reduced from 5e-4 (v2) to 2e-4 (v3) for more stable training
  - Early stopping patience was increased from 15 to 20 epochs to allow the slow GRL ramp to take full effect

- **Code attribution:** PyTorch framework, GRL implementation adapted from Ganin & Lempitsky (2015) description. No external code copied directly.
