---
{}
---

# Model Card for AV-StyleLGBM: LightGBM with Comprehensive Stylometric Features

A LightGBM gradient boosting classifier for authorship verification (AV) that determines whether two texts were written by the same author, using ~695 stylometric features per text pair including novel syntactic complexity profiling, writing rhythm analysis, and information-theoretic authorial signatures.

## Model Details

### Model Description

AV-StyleLGBM is a Category A (traditional ML) solution for the COMP34812 Authorship Verification shared task. Given a pair of texts, it extracts comprehensive stylometric features from each text independently, computes element-wise absolute difference vectors (diff-vectors) to capture authorial style divergence, and classifies the pair as same-author (1) or different-author (0) using LightGBM.

The key innovations are:
1. **Comprehensive feature engineering** spanning 9 feature groups (435 features per text), including novel syntactic complexity profiling, writing rhythm analysis, and information-theoretic authorial signatures
2. **Topic-robustness mechanism** via function-word-only style diff-vectors and topic-correlated feature filtering, addressing the style-content confound without neural adversarial training
3. **LightGBM classifier** with tuned hyperparameters for optimal gradient boosting on high-dimensional stylometric feature spaces

- **Developed by:** Group 34
- **Language(s):** English
- **Model type:** Gradient boosting classifier (traditional ML, Category A)
- **Model architecture:** StandardScaler -> LGBMClassifier (1000 trees, max_depth=7, 63 leaves)
- **Finetuned from model [optional]:** N/A (trained from scratch)

### Model Resources

- **Repository:** https://github.com/ayushkumarcode/comp34812-nlu-coursework
- **Paper or documentation:**
  - Stamatatos et al. (2023) "Same or Different? Diff-Vectors for Authorship Analysis" -- diff-vector representation
  - Abbasi & Chen (2008) "Writeprints" -- baseline feature set (extended with novel Groups 7-9)
  - Jiang et al. (2023) "Low-Resource Text Classification: A Parameter-Free Classification Method with Compressors" -- NCD features
  - Bevendorff et al. (2022) PAN 2022 -- character n-gram cosine similarity baseline
  - Ke et al. (2017) "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" -- LightGBM classifier
  - Burrows (2002) "Delta" -- stylometric distance with stability guard for short texts

## Training Details

### Training Data

- **Dataset:** COMP34812 AV shared task training data
- **Size:** 27,643 text pairs
- **Label distribution:** ~50% same-author (class 1), ~50% different-author (class 0)
- **Text sources:** Enron emails, blog posts, movie reviews (cross-domain)
- **Preprocessing:** HTML entity decoding, Unicode NFC normalization, URL replacement with `<URL>` token. Case and punctuation preserved (stylistic signals). No stemming or lemmatization.

### Training Procedure

#### Feature Engineering Pipeline

1. **Per-text features (435 features per text):**
   - Group 1: Lexical features (29) -- word length distribution, vocabulary richness (TTR, Yule's K, Simpson's D, Honore's R, Brunet's W)
   - Group 2: Character-level features (56) -- letter, digit, and special character frequency distributions
   - Group 3: Character n-gram TF-IDF + SVD (100) -- char (3,5)-gram TF-IDF reduced to 100 dimensions via TruncatedSVD
   - Group 4: Function word frequencies (169) -- normalized frequencies of 169 English function words
   - Group 5: POS tag features (45) -- 17 universal POS tag frequencies + 28 POS bigram frequencies (spaCy en_core_web_md)
   - Group 6: Structural features (15) -- sentence length statistics, paragraph structure, punctuation density, capitalization
   - Group 7: Syntactic complexity features (10, NOVEL) -- dependency parse depth, branching factor, subordination index, arc length, passive constructions, relative clauses, coordination complexity, content clauses, fronted adverbials
   - Group 8: Writing rhythm features (6, NOVEL) -- sentence length autocorrelation, sentence length entropy, punctuation burstiness, variance ratio, mean-reversion tendency, punctuation diversity entropy
   - Group 9: Information-theoretic features (5, NOVEL) -- character bigram mutual information, text entropy rate, conditional entropy, word length entropy, rolling TTR entropy

2. **Diff-vector computation:** |f(text_1) - f(text_2)| for all per-text features (~435 dims)

3. **Style-only diff-vector (NOVEL topic-robustness):** Separate diff-vector using only function words, POS tags, syntactic, rhythm, and information-theoretic features (~246 dims)

4. **Pairwise features (14):** NCD (gzip, lzma, bz2), cosine similarity of character 3/4/5-gram TF-IDF vectors, Jensen-Shannon divergence (word, char bigram), Jaccard overlap, content word overlap, length/word count/vocabulary ratios, Burrows' Delta (with 200-word stability guard)

5. **Total feature vector: ~695 features per pair**

#### Training Hyperparameters

- **Classifier:** LGBMClassifier
  - n_estimators: 1000
  - max_depth: 7
  - learning_rate: 0.05
  - num_leaves: 63
  - subsample: 0.8
  - colsample_bytree: 0.8
  - min_child_samples: 20
  - reg_alpha: 0.1
  - reg_lambda: 1
  - random_state: 42
- **Feature scaling:** StandardScaler (fitted on training data only)

#### Speeds, Sizes, Times

- **Feature extraction:** ~15-30 minutes on CPU (27K pairs with spaCy processing)
- **Model training:** ~2-5 minutes on CPU
- **Total training time:** ~20-35 minutes
- **Model size:** ~14.6MB (5 files: LightGBM model + scaler + TF-IDF vectorizers + feature names)
- **Inference speed:** ~100 pairs/second (bottleneck is spaCy feature extraction)

## Evaluation

### Testing Data & Metrics

#### Testing Data

- **Dev set:** COMP34812 AV dev set (5,993 evaluation pairs)
- **Class distribution:** ~50/50 same-author vs different-author

#### Metrics

- **Primary:** macro_f1 (macro-averaged F1 score)
- **Secondary:** accuracy, Matthews Correlation Coefficient (MCC)
- **Statistical tests:** McNemar's test (vs baselines), bootstrap 95% confidence intervals (1000 iterations)

### Results

| Metric | Value |
|--------|-------|
| macro_f1 | 0.7340 |
| accuracy | 0.7340 |
| MCC | 0.4690 |

**Baseline comparison:**

| Model | macro_f1 | vs Ours (gap) |
|-------|----------|---------------|
| SVM baseline | 0.5610 | +0.1730 |
| LSTM baseline | 0.6226 | +0.1114 |
| BERT baseline | 0.7854 | -0.0514 |

The model significantly outperforms both the SVM and LSTM baselines. While it does not surpass the BERT baseline, it achieves competitive performance using only handcrafted features and a traditional ML classifier, demonstrating the value of comprehensive stylometric analysis.

## Technical Specifications

### Hardware

- **Training:** CSF3 HPC cluster, CPU nodes (no GPU required for Category A)
- **Inference:** Any machine with Python 3.11+, ~4GB RAM

### Software

- Python 3.11
- scikit-learn >= 1.3
- lightgbm >= 4.0
- spaCy >= 3.7 with en_core_web_md model
- numpy, pandas, joblib

## Bias, Risks, and Limitations

- **Domain sensitivity:** Performance may degrade on text domains not represented in training (e.g., scientific papers, social media posts). The model was trained on Enron emails, blog posts, and movie reviews.
- **Text length sensitivity:** Vocabulary richness features (Yule's K, Honore's R) are imputed for texts with fewer than 50 words, and Burrows' Delta requires 200+ words per text. Very short texts may receive degraded predictions.
- **Topic confound:** Despite the topic-robustness mechanism, same-topic different-author pairs may still be misclassified if authors share similar writing styles. The style-only diff-vector mitigates but does not eliminate this.
- **Language assumption:** Features are designed for English text only. Function words, POS tags, and syntactic features assume English grammar.
- **Authorship vs style:** The model detects stylistic similarity, not true authorship. Writers who deliberately imitate another's style could fool the model.

## Additional Information

- **Novel contributions:**
  1. Syntactic complexity profiling for AV (dependency depth, subordination, arc length) -- inspired by Feng et al. (2012) on syntactic stylometry
  2. Writing rhythm analysis (sentence-length autocorrelation, punctuation burstiness) -- time-series analysis applied to linguistic sequences
  3. Information-theoretic authorial signatures (char bigram MI, entropy rate, conditional entropy) -- finer-grained than NCD
  4. Topic-robustness mechanism for Cat A (style-only diff-vector, function-word subspace) -- addresses style-content confound without neural adversarial training

- **Code attribution:** scikit-learn StandardScaler, LightGBM classifier, spaCy NLP pipeline, standard Python libraries. No external code copied directly.
