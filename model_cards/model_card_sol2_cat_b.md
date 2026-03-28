---
{}
---

# Model Card for AV-StyleDisentangle: Adversarial Style-Content Disentanglement Network

A Siamese neural network for authorship verification that learns to separate writing style from topic content. It uses gradient reversal adversarial training, character-level CNN + BiLSTM encoding, and additive attention to produce style-focused text representations.

## Model Details

### Model Description

AV-StyleDisentangle is our Category B (neural network, no pre-trained transformers) solution for the COMP34812 Authorship Verification shared task. Each text gets fed through a shared (Siamese) encoder as a character sequence: multi-width character-level CNNs (kernels 3, 5, 7) followed by a BiLSTM and additive (Bahdanau) attention, producing a 128-dimensional document embedding. We then compare two text embeddings by concatenating [v1, v2, |v1-v2|, v1*v2] (512 dimensions total) and running that through an MLP for classification.

The main idea here is adversarial style-content disentanglement: we attach a gradient reversal layer (GRL) to an auxiliary topic prediction head, which forces the encoder to learn representations that aren't predictive of topic/domain. This means it focuses on authorial style instead. It's designed to solve a real problem in cross-domain AV where same-topic pairs get misclassified as same-author just because they're about the same thing.

We also added a few other things that helped:
1. **Stylistic invariance training** through character perturbation augmentation (5% per-char) and random truncation (80-100%)
2. **Careful GRL scheduling** -- lambda ramps linearly from 0 to 0.05 over 20 epochs, and we don't introduce the topic adversarial loss until epoch 15 (we tried earlier and it destabilized training)
3. **Interpretable attention weights** via Bahdanau attention, giving us some XAI explainability

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
- **Preprocessing:** Character-level encoding (vocabulary size 97: a-z, A-Z, 0-9, common punctuation, whitespace, UNK, PAD). We didn't do any text normalization beyond replacing URLs. Max sequence length is 1500 characters.
- **Topic pseudo-labels:** We generated these via TF-IDF (5000 features) + MiniBatch K-Means clustering (10 clusters) on all training texts. We also used heuristic corpus-type labels (email/blog/review/unknown) when coverage was good enough.

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
- **Loss:** BCEWithLogitsLoss + topic adversarial CrossEntropyLoss (we dropped contrastive loss -- see design decisions below)
- **Topic adversarial weight:** 0.02 (only applied from epoch 15 onward)
- **GRL schedule:** lambda ramps linearly from 0 to 0.05 over epochs 1-20, then stays fixed at 0.05
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

We beat both the SVM and LSTM baselines by a solid margin. It doesn't quite reach BERT, but for a model that's trained entirely from scratch on raw characters with no pre-trained language representations, we're pretty happy with 0.7422. The gap to BERT is only about 4 points, and the adversarial topic debiasing does seem to help with cross-domain robustness.

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

- **Character-level representation:** Since we operate on raw characters without any linguistic preprocessing, we preserve stylistic signals like spacing, punctuation, and casing. The downside is that OCR artifacts or encoding inconsistencies could throw things off.
- **Domain shift:** The GRL topic debiasing helps with cross-domain generalization, but it probably won't fully transfer to completely unseen domains like legal documents or code-switching texts.
- **Sequence length:** We truncate to 1500 characters, so any stylistic patterns near the end of very long texts get lost. We tried longer sequences but it didn't improve results enough to justify the memory cost.
- **Attention interpretability:** The attention weights give us some window into what the model focuses on, but they shouldn't be taken as definitive feature importance scores (Jain & Wallace, 2019).
- **Topic cluster quality:** Our topic pseudo-labels come from TF-IDF + KMeans, which isn't perfect at capturing domain boundaries. This could limit how well the adversarial debiasing works in practice.

## Additional Information

- **Novel contributions:**
  1. Adversarial style-content disentanglement via GRL for AV -- we adapted Ganin & Lempitsky (2015)'s domain adaptation idea to the authorship verification setting
  2. Careful GRL scheduling (slow lambda ramp over 20 epochs, delayed topic loss from epoch 15) -- we found this was essential to prevent training instability
  3. Stylistic invariance training via character perturbation augmentation
  4. Interpretable attention weights (XAI contribution via Bahdanau attention)

- **Design decisions:**
  - We tried contrastive loss in earlier versions (v1, v2) but ended up dropping it in the final model (v3) because BCE + topic adversarial loss alone actually performed better
  - Learning rate went from 5e-4 (v2) down to 2e-4 (v3) for more stable training -- the higher rate caused some oscillation
  - We bumped early stopping patience from 15 to 20 epochs so the slow GRL ramp has time to take full effect before we give up

- **Code attribution:** PyTorch framework, GRL implementation adapted from Ganin & Lempitsky (2015) description. No external code copied directly.
