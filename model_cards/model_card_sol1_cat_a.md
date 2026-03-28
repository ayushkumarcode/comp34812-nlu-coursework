---
{}
---

# Model Card for AV-StyleLGBM: LightGBM with Comprehensive Stylometric Features

A LightGBM gradient boosting classifier for authorship verification (AV) that figures out whether two texts were written by the same author. It uses ~736 stylometric features per text pair, including syntactic complexity profiling, writing rhythm analysis, information-theoretic authorial signatures, FFT spectral analysis, Zipf-Mandelbrot law deviation, Benford's law analysis, and fractal/Hurst exponent features that we designed ourselves.

## Model Details

### Model Description

AV-StyleLGBM is our Category A (traditional ML) solution for the COMP34812 Authorship Verification shared task. Given a pair of texts, we extract a big set of stylometric features from each text independently, compute element-wise absolute difference vectors (diff-vectors) to capture how much the authorial styles diverge, and then classify the pair as same-author (1) or different-author (0) with LightGBM.

What makes this approach work is three things:
1. **Comprehensive feature engineering** across 13 feature groups (456 features per text), including novel syntactic complexity profiling, writing rhythm analysis, information-theoretic authorial signatures, FFT spectral analysis of sentence rhythms, Zipf-Mandelbrot law deviation, Benford's law on linguistic distributions, and fractal analysis via Hurst exponents
2. **A topic-robustness mechanism** using function-word-only style diff-vectors and topic-correlated feature filtering -- this tackles the style-content confound without needing neural adversarial training
3. **LightGBM classifier** with tuned hyperparameters (2000 trees, depth 8) that handles the high-dimensional stylometric feature space well

- **Developed by:** Group 34
- **Language(s):** English
- **Model type:** Gradient boosting classifier (traditional ML, Category A)
- **Model architecture:** StandardScaler -> LGBMClassifier (2000 trees, max_depth=8, 127 leaves)
- **Finetuned from model [optional]:** N/A (trained from scratch)

### Model Resources

- **Repository:** https://github.com/ayushkumarcode/comp34812-nlu-coursework
- **Paper or documentation:**
  - Stamatatos et al. (2023) "Same or Different? Diff-Vectors for Authorship Analysis" -- diff-vector representation
  - Abbasi & Chen (2008) "Writeprints" -- baseline feature set (we extended it with our novel Groups 7-13)
  - Jiang et al. (2023) "Low-Resource Text Classification: A Parameter-Free Classification Method with Compressors" -- NCD features
  - Bevendorff et al. (2022) PAN 2022 -- character n-gram cosine similarity baseline
  - Ke et al. (2017) "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" -- LightGBM classifier
  - Burrows (2002) "Delta" -- stylometric distance with stability guard for short texts
  - Evert et al. (2017) "Understanding and explaining Delta measures for authorship attribution" -- Cosine Delta distance
  - Ausloos (2012) "Generalized Hurst exponent and multifractal function of original and translated texts" Phys. Rev. E -- Hurst exponent for text analysis
  - Ausloos (2014) "Measuring complexity with multifractals in texts. Translation effects" -- Zipf-Mandelbrot law fitting for style
  - Moreno-Sanchez et al. (2016) "Large-Scale Analysis of Zipf's Law in English Texts" PLOS ONE -- Zipf law parameter estimation
  - Sambridge & Tkalcic (2010) "Benford's Law of First Digits: A Universal Phenomenon" -- Benford's law as a fingerprint for data distributions

## Training Details

### Training Data

- **Dataset:** COMP34812 AV shared task training data
- **Size:** 27,643 text pairs
- **Label distribution:** ~50% same-author (class 1), ~50% different-author (class 0)
- **Text sources:** Enron emails, blog posts, movie reviews (cross-domain)
- **Preprocessing:** HTML entity decoding, Unicode NFC normalization, URL replacement with `<URL>` token. We kept case and punctuation intact since they're important stylistic signals. No stemming or lemmatization.

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

3. **Style-only diff-vector (NOVEL topic-robustness):** We compute a separate diff-vector using only function words, POS tags, syntactic, rhythm, and information-theoretic features (~246 dims). The idea is to strip out anything that might be topic-related and focus purely on style.

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
- **Inference speed:** ~100 pairs/second (the bottleneck is spaCy feature extraction)

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

We significantly outperform both the SVM and LSTM baselines. It doesn't beat BERT, but we think it's still a strong result for a purely handcrafted-feature approach -- getting within 5 points of a fine-tuned transformer with no pre-trained representations at all shows that thoughtful stylometric analysis still has real value.

## Technical Specifications

### Hardware

- **Training:** CSF3 HPC cluster, CPU nodes (no GPU needed for Category A)
- **Inference:** Any machine with Python 3.11+, ~4GB RAM

### Software

- Python 3.11
- scikit-learn >= 1.3
- lightgbm >= 4.0
- spaCy >= 3.7 with en_core_web_md model
- numpy, pandas, joblib

## Bias, Risks, and Limitations

- **Domain sensitivity:** Performance will likely drop on text types we didn't train on (e.g., scientific papers, social media posts). Our training data covers Enron emails, blog posts, and movie reviews, so anything outside those domains is a risk.
- **Text length sensitivity:** Some vocabulary richness features (Yule's K, Honore's R) get imputed for texts under 50 words, and Burrows' Delta needs 200+ words per text to be reliable. Very short texts won't get the best predictions.
- **Topic confound:** Even with the topic-robustness mechanism, same-topic different-author pairs can still trip up the model if the authors happen to write similarly. The style-only diff-vector helps but doesn't completely solve this.
- **Language assumption:** Everything's designed for English only. The function words, POS tags, and syntactic features all assume English grammar.
- **Authorship vs style:** We're really detecting stylistic similarity, not proving authorship. Someone deliberately imitating another writer's style could fool the model.

## Additional Information

- **Novel contributions:**
  1. Syntactic complexity profiling for AV (dependency depth, subordination, arc length) -- we drew on Feng et al. (2012)'s work on syntactic stylometry
  2. Writing rhythm analysis (sentence-length autocorrelation, punctuation burstiness) -- basically applying time-series analysis to linguistic sequences, which turned out to capture authorial habits pretty well
  3. Information-theoretic authorial signatures (char bigram MI, entropy rate, conditional entropy) -- these give a finer-grained picture than NCD alone
  4. Topic-robustness mechanism for Cat A (style-only diff-vector, function-word subspace) -- our way of handling the style-content confound without needing any neural adversarial training

- **Code attribution:** scikit-learn StandardScaler, LightGBM classifier, spaCy NLP pipeline, standard Python libraries. No external code copied directly.
