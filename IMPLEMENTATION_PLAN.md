Good. 5993 rows in the solution (no header) and 5994 in the baseline CSV (1 header + 5993 data rows). The dev set has 6004 data rows in dev.csv (6005 - 1 header), but the solution has 5993. That is a discrepancy worth noting -- 11 rows less, which aligns with the ~10.5% duplicate overlap mentioned. Actually that suggests the dev evaluation excludes some samples, or the prediction file simply needs to match the solution file row count.

Now I have all the information needed to create the exhaustive implementation plan. Let me synthesize everything.

---

# EXHAUSTIVE IMPLEMENTATION PLAN: COMP34812 Authorship Verification (AV Track) -- Targeting 95-100%

## STRATEGIC OVERVIEW

**Track:** Authorship Verification (AV)
**Strategy:** Build ALL THREE category solutions (A, B, C), then select the best two for submission. This maximizes optionality and de-risks the project.

**Submission constraints verified from the coursework spec and scorer:**
- Prediction files: `Group_n_A.csv`, `Group_n_B.csv`, or `Group_n_C.csv` -- single column of integers (0 or 1), no header needed (the scorer auto-skips a non-numeric header if present), one value per line
- Dev set: 5993 evaluation rows (from `NLU_SharedTask_AV_dev.solution`); train set: 27,643 rows
- Metrics: accuracy, macro precision, macro recall, **macro_f1** (primary), weighted variants, **matthews_corrcoef** (MCC) -- all computed by the local scorer at `/Users/kumar/Documents/University/Year3/NLU/project/baseline_extracted/nlu_bundle-feature-unified-local-scorer/`
- Baselines to beat: SVM macro_f1=0.5610/MCC=0.1235; LSTM macro_f1=0.6226/MCC=0.2452; BERT macro_f1=0.7854/MCC=0.5709
- **Closed mode**: only provided training data allowed, no external datasets

---

## SOLUTION 1: CATEGORY A -- Diff-Vector Stacking Ensemble with Comprehensive Stylometrics

### 1.1 Architecture Overview

Extract ~400 stylometric features per text. For each pair (text_1, text_2), compute the absolute element-wise difference vector |f(text_1) - f(text_2)| of dimension ~400. Additionally append ~10 pairwise similarity features (NCD, cosine similarity, etc.) for a total input dimension of ~410. Feed into a stacking ensemble: base classifiers (SVM-RBF, Random Forest, XGBoost) whose predictions are meta-learned by a Logistic Regression classifier.

### 1.2 Data Preprocessing

1. **Load CSV** via pandas with proper quoting handling (`quotechar='"'`, `escapechar=None`, use Python csv engine). The training data at `/Users/kumar/Documents/University/Year3/NLU/project/training_extracted/training_data/AV/train.csv` uses standard CSV quoting with commas inside quoted fields.

2. **Text cleaning** (minimal, to preserve stylistic signal):
   - Decode HTML entities (`&amp;` -> `&`, etc.)
   - Normalize Unicode (NFC normalization)
   - Strip leading/trailing whitespace
   - Do NOT lowercase (case patterns are stylometric features)
   - Do NOT remove punctuation (punctuation patterns are features)
   - Do NOT stem/lemmatize (morphological patterns are features)
   - Replace URLs with a single token `<URL>` (URLs are not authorial style, and they inflate character n-gram noise)

3. **Label handling**: Labels in train.csv are floats (0.0, 1.0). Convert to int.

4. **Train/dev split**: Use the provided split. For internal validation during development, use 5-fold stratified CV on training data.

### 1.3 Feature Engineering (Per-Text, ~400 Features)

**Group 1: Lexical Features (30 features)**
- Average word length
- Word length distribution: proportion of words of length 1, 2, 3, ..., 20 (20 features)
- Vocabulary richness metrics:
  - Type-Token Ratio (TTR)
  - Hapax legomena ratio (words appearing exactly once / total words)
  - Hapax dis legomena ratio (words appearing exactly twice / total words)
  - Yule's K measure: `K = 10^4 * (sum_i(i^2 * V_i) - N) / N^2` where V_i = number of words occurring i times, N = total words
  - Simpson's Diversity Index: `D = 1 - sum(n_i * (n_i - 1)) / (N * (N-1))`
  - Honore's R: `R = 100 * log(N) / (1 - V_1/V)` where V_1 = hapax, V = vocabulary size
  - Brunet's W: `W = N^(V^-0.172)`
- Total word count (as a feature, not just for normalization)

**Group 2: Character-Level Features (26 + 10 + 20 = 56 features)**
- Letter frequency distribution (a-z, case-insensitive, normalized by total characters): 26 features
- Digit frequency distribution (0-9, normalized): 10 features
- Special character frequencies (each normalized by total characters): `.` `,` `;` `:` `!` `?` `'` `"` `-` `(` `)` `/` `\` `@` `#` `$` `%` `&` `*` `_`: 20 features

**Group 3: Character N-gram TF-IDF (dimensionality-reduced, ~100 features)**
- Fit a TF-IDF vectorizer on ALL training texts (both text_1 and text_2 columns) with `analyzer='char'`, `ngram_range=(3,5)`, `max_features=10000`
- Apply TruncatedSVD to reduce to 100 dimensions per text (100 is a starting point; experiment with [50, 100, 200] and select via CV)
- This captures the character n-gram signature of each text in a dense representation
- **Critical implementation detail**: Fit the TF-IDF vectorizer and SVD on training data only, then transform dev/test data

**Group 4: Function Word Frequencies (150 features)**
- Use a standard list of 150 English function words (pronouns, prepositions, conjunctions, auxiliary verbs, determiners, etc.): the, of, and, a, to, in, is, it, that, was, for, on, are, with, as, I, his, they, be, at, one, have, this, from, or, had, by, but, not, what, all, were, we, when, your, can, said, there, each, which, she, do, how, their, if, will, up, about, out, many, then, them, these, so, some, her, would, make, like, into, him, has, two, more, no, way, could, my, than, first, been, who, its, now, over, just, other, also, after, very, because, before, however, most, should, where, still, must, while, therefore, although, since, during, until, unless, instead, neither, either, moreover, furthermore, consequently, nevertheless, thus, hence, accordingly, yet, though, whether, per, both, such, those, any, own, an, only, being, did, another, may, might, shall, upon, much, often, perhaps, again, too, once, already, above, below, between, through, around, against, without, within, along, among, across, toward, including, during, following, according, regarding, concerning, despite, except, beyond, besides, underneath, alongside, outside, inside, throughout, beneath, behind, ahead, apart, aside, away, everywhere, nowhere, somehow, somewhat, sometimes, somewhere, otherwise
- Compute frequency of each function word normalized by total word count

**Group 5: POS Tag Features (45 features)**
- Use spaCy (`en_core_web_md`) for POS tagging (prefer `en_core_web_md` over `en_core_web_sm` for better POS accuracy on informal text such as emails and blogs)
- Universal POS tag frequencies (17 tags: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X): 17 features
- POS bigram frequencies (top 28 most frequent POS bigrams from training data): 28 features

**Group 6: Structural Features (15 features)**
- Average sentence length (words per sentence, using spaCy sentence segmentation)
- Sentence length standard deviation
- Median sentence length
- Max sentence length
- Min sentence length
- Number of sentences
- Percentage of short sentences (< 8 words)
- Percentage of long sentences (> 25 words)
- Average word count per paragraph (split on `\n\n`)
- Number of paragraphs
- Ratio of text enclosed in quotes/dialogue
- Exclamation density (exclamation marks per sentence)
- Question density (question marks per sentence)
- Ellipsis count (normalized by sentence count)
- Capitalization ratio (uppercase letters / total letters)

**Group 7: Syntactic Complexity Features (10 features) -- NOVEL for AV**
These features go beyond standard POS-tag frequency analysis used in PAN methods. They capture deep syntactic patterns that are highly author-specific and rarely exploited in AV literature.
- Average dependency parse depth (mean depth of dependency trees across sentences, via spaCy `en_core_web_md`)
- Maximum dependency parse depth (captures most complex sentence)
- Average branching factor of dependency trees (mean number of children per non-leaf node)
- Subordination index: subordinate clause count per sentence (count of SCONJ-headed clauses)
- Average dependency arc length (mean absolute distance between head and dependent token positions)
- Proportion of passive constructions (detect via auxiliary "be" + past participle dependency pattern)
- Relative clause count per sentence (normalized count of "relcl" dependency relations)
- Mean number of conjuncts per coordination (captures coordination complexity)
- Ratio of content clauses ("ccomp" + "xcomp" deps per sentence)
- Proportion of fronted adverbials (adverbial modifiers before main verb)

**Group 8: Writing Rhythm Features (6 features) -- NOVEL for AV**
These features capture temporal/sequential patterns in writing that are unconscious authorial habits, not captured by aggregate statistics.
- Sentence length sequence autocorrelation (lag-1 autocorrelation of sentence-length sequence: do they alternate short/long or cluster similar lengths?)
- Entropy of sentence length distribution (Shannon entropy of binned sentence lengths -- higher means more varied rhythm)
- Burstiness of punctuation patterns: coefficient of variation of inter-punctuation distances (how regularly/irregularly an author punctuates)
- Sentence length variance ratio: variance of first-half sentence lengths vs second-half (do authors fatigue / change rhythm?)
- Mean-reversion tendency of sentence lengths (does the author return to a baseline sentence length after deviating?)
- Punctuation diversity entropy: Shannon entropy over punctuation type frequencies

**Group 9: Information-Theoretic Features (5 features) -- NOVEL for AV**
These features capture the information-theoretic signature of an author's character/word usage patterns, going beyond simple NCD.
- Character bigram mutual information: average pointwise mutual information across all character bigram types observed in the text
- Text entropy rate: per-character Shannon entropy of the text (bits per character)
- Conditional entropy of character sequences: H(c_n | c_{n-1}) estimated from bigram and unigram counts
- Word length entropy: Shannon entropy of the word-length frequency distribution
- Type-token ratio entropy: rolling TTR entropy (compute TTR in sliding windows of 50 words, then take entropy of the TTR distribution)

**Total per text: ~468 features (396 original + 10 syntactic + 6 rhythm + 5 info-theoretic + ~51 function-word-only features from topic-robustness mechanism below)**

### 1.3b Topic-Robustness Mechanism for Category A (NOVEL)

Category B has GRL for topic debiasing, but Category A currently lacks an explicit topic-robustness mechanism. Add the following:

1. **Function-word-only feature subspace**: Compute a SEPARATE set of ~51 features using ONLY function words (all lexical features, vocabulary richness, and structural features computed after stripping content words). This creates a purely stylistic feature set immune to topic confounds.

2. **Topic-correlated feature penalty**: During feature selection, compute the Pearson correlation between each feature and a topic proxy (the K-means cluster assignment from TF-IDF of content words). Penalize or drop features with |correlation| > 0.3 with topic. This can be implemented as a custom scoring function in sklearn's `SelectKBest`.

3. **Diff-vector on style-sensitive features only**: In addition to the full diff-vector, compute a SEPARATE diff-vector using only function word frequencies, POS tags, syntactic complexity, writing rhythm, and information-theoretic features (dropping character n-gram TF-IDF and content-word-based features). Feed BOTH diff-vectors to the ensemble.

### 1.4 Pairwise Features (Appended to Diff-Vector)

Beyond the element-wise absolute difference |f(text_1) - f(text_2)|, append these pairwise similarity/distance features (~14 features):

1. **Normalized Compression Distance (NCD) -- gzip**: `NCD = (C(t1+t2) - min(C(t1), C(t2))) / max(C(t1), C(t2))` where C(x) = `len(gzip.compress(x.encode('utf-8')))`. Cite: Jiang et al. (2023) "Low-Resource Text Classification: A Parameter-Free Classification Method with Compressors", ACL Findings.
2. **NCD -- lzma**: Same formula, using `lzma.compress`. Second compressor provides orthogonal signal.
3. **NCD -- bz2**: Third compressor variant.
4. **Cosine similarity of character 3-gram TF-IDF vectors**: Compute raw TF-IDF vectors for each text (not the SVD-reduced ones), then cosine similarity. This is the PAN 2022 baseline that beat all DL systems.
5. **Cosine similarity of character 4-gram TF-IDF vectors**
6. **Cosine similarity of character 5-gram TF-IDF vectors**
7. **Jensen-Shannon divergence of word frequency distributions**: Compute word frequency distributions for each text, then JSD.
8. **Jensen-Shannon divergence of character bigram distributions**
9. **Word overlap ratio**: |words_1 intersection words_2| / |words_1 union words_2| (Jaccard index)
10. **Content word overlap**: Same but only for non-function-words
11. **Length ratio**: len(text_1) / len(text_2) (or min/max to normalize)
12. **Word count ratio**: word_count_1 / word_count_2
13. **Vocabulary size ratio**: vocab_1 / vocab_2
14. **Burrows' Delta**: Classic stylometric distance using z-scored word frequencies of most frequent words. Cite: Burrows (2002). **IMPORTANT GUARD**: Only compute when BOTH texts have >= 200 words. On shorter texts, Burrows' Delta is mathematically unstable (z-scores of word frequencies become highly noisy with small samples). If either text has < 200 words, impute with the training set median Burrows' Delta value.

**Final feature vector dimension: ~468 (diff-vector from all per-text features) + ~468 (style-only diff-vector) + 14 (pairwise) = ~950**
**Note**: The exact dimension will depend on feature selection outcomes. The style-only diff-vector may be smaller (~250 features) after dropping content-correlated features.

### 1.5 Classifier: Stacking Ensemble

**Base classifiers (Level 0):**

1. **SVM-RBF**:
   - Preprocessing: StandardScaler (zero mean, unit variance)
   - Hyperparameter search (GridSearchCV, 5-fold stratified):
     - C: [0.01, 0.1, 1, 10, 100]
     - gamma: ['scale', 'auto', 0.001, 0.01, 0.1]
   - Use `probability=True` for soft predictions to feed into meta-learner

2. **Random Forest**:
   - No scaling needed
   - Hyperparameter search:
     - n_estimators: [200, 500, 1000]
     - max_depth: [10, 20, 30, None]
     - min_samples_split: [2, 5, 10]
     - min_samples_leaf: [1, 2, 4]
     - class_weight: ['balanced', None]
   - Feature importance extraction for analysis

3. **XGBoost**:
   - No scaling needed
   - Hyperparameter search:
     - n_estimators: [200, 500, 1000]
     - max_depth: [3, 5, 7, 10]
     - learning_rate: [0.01, 0.05, 0.1]
     - subsample: [0.7, 0.8, 0.9]
     - colsample_bytree: [0.7, 0.8, 0.9]
     - reg_alpha: [0, 0.1, 1]
     - reg_lambda: [1, 5, 10]

**Meta-classifier (Level 1): Logistic Regression**
- Input: probability predictions from all 3 base classifiers (3 features if binary probs, 6 if using predict_proba for both classes)
- Use sklearn `StackingClassifier` with `cv=5` and `passthrough=False`
- Alternative: also pass the original features through (`passthrough=True`) and let LogReg select
- Regularization: C=1.0, solver='lbfgs'

**Training procedure:**
- The stacking classifier uses internal 5-fold CV to generate out-of-fold predictions from base classifiers, then trains the meta-learner on those predictions
- Final evaluation on the held-out dev set
- Save all fitted models using `joblib.dump`

### 1.6 Creativity Justification (for Model Card and Presentation)

**Central novel contributions (these are the creativity differentiators, NOT standard PAN methodology):**

1. **Syntactic complexity profiling for AV (NOVEL)**: Standard PAN AV systems use surface-level POS tag frequencies. Our system goes deeper: dependency parse depth, branching factor, subordination index, arc length, and passive construction ratios capture syntactic habits that are unconscious and highly author-specific. This is inspired by computational stylistics research (e.g., Feng et al. 2012 on syntactic stylometry) but rarely applied in modern AV systems.

2. **Writing rhythm analysis (NOVEL)**: Sentence-length autocorrelation and burstiness of punctuation patterns capture temporal writing habits -- whether an author alternates short/long sentences, maintains regular punctuation intervals, or varies rhythmically. This is a genuinely novel feature category for AV, drawing from time-series analysis applied to linguistic sequences.

3. **Information-theoretic authorial signatures (NOVEL beyond NCD)**: While NCD is established, we additionally compute character bigram mutual information, text entropy rate, and conditional entropy of character sequences. These capture the information-theoretic fingerprint of an author's character usage at a finer grain than compression-based distance.

4. **Topic-robustness mechanism (NOVEL for Cat A)**: Unlike typical Cat A systems that are vulnerable to topic confounds, we implement explicit topic debiasing: function-word-only feature subspace, topic-correlated feature penalty, and a separate style-only diff-vector. This is the first Cat A AV system (to our knowledge) that systematically addresses the style-content confound without neural adversarial training.

**Standard methodology (foundations):**
- **Diff-vector representation**: Cite Stamatatos et al. (2023) "Same or Different? Diff-Vectors for Authorship Analysis"
- **Writeprints feature set (baseline)**: Cite Abbasi & Chen (2008) -- our system extends this significantly with Groups 7-9
- **NCD as a feature**: Cite Jiang et al. (2023), the "gzip classifier" paper
- **Character n-gram cosine similarity**: Cite Bevendorff et al. (2022) PAN 2022 overview
- **Stacking ensemble**: Cite Wolpert (1992) and Weerasinghe & Greenstadt (2020) PAN 2020
- **Burrows' Delta (with stability guard)**: Cite Burrows (2002)

### 1.7 Soundness Checklist

- Feature scaling applied before SVM (StandardScaler fitted on train only)
- TF-IDF and SVD fitted on train only, transform applied to dev/test
- No data leakage: dev set never seen during training/feature fitting
- Function word list is language-appropriate (English)
- **Vocabulary richness guards (EXACT specification)**: If `word_count < 50`, impute ALL vocabulary richness features (TTR, hapax ratio, Yule's K, Simpson's D, Honore's R, Brunet's W) with the training set median for that feature. Honore's R is mathematically undefined when V_1 = V (all words are hapax legomena), which happens frequently in very short texts -- the guard prevents division-by-zero / log-domain errors. Additionally, for Yule's K, guard against N < 2 (returns 0).
- Stacking uses internal CV, not test/dev data for meta-learner training
- Class balance is verified (~50/50, no special handling needed)
- All hyperparameter selection uses cross-validation on training data

---

## SOLUTION 2: CATEGORY B -- Adversarial Style-Content Disentanglement Network (Siamese Character-CNN + BiLSTM + Attention)

### 2.1 Architecture Overview

An Adversarial Style-Content Disentanglement Network for authorship verification. The CENTRAL innovation is the use of gradient reversal to force the shared encoder to produce style representations explicitly disentangled from topic/content -- addressing the key confound in cross-domain AV where same-topic pairs are easily misclassified as same-author. The architecture is a Siamese neural network operating on character-level input, where each text is processed through a shared encoder consisting of character embeddings, parallel multi-width Conv1D layers, a BiLSTM, and additive attention, producing a fixed-size document embedding. The two embeddings are compared via concatenation of [v1, v2, |v1-v2|, v1*v2], fed through an MLP classifier. A gradient reversal layer (GRL) on an auxiliary domain prediction head, combined with contrastive learning at the embedding level, ensures the learned representations capture authorial style rather than topic.

**Key architectural contributions:**
1. **Adversarial style-content disentanglement via GRL** (central contribution)
2. **Contrastive embedding loss** as a primary training objective (not optional)
3. **Stylistic invariance training** via character perturbation augmentation
4. **Interpretable attention weights** providing explainability (XAI contribution)

### 2.2 Data Preprocessing

1. **Character vocabulary**: Build from training data. Include: lowercase a-z (26), uppercase A-Z (26), digits 0-9 (10), common punctuation and symbols (`.,:;!?'"()-/\@#$%&*_+=<>[]{}|~` -- approximately 30), whitespace (space, tab, newline -- 3 tokens), and a special `<UNK>` token and `<PAD>` token. Total vocabulary: ~97 characters.

2. **Character encoding**: Map each character to an integer index. Replace any character not in vocabulary with `<UNK>` index.

3. **Sequence length**: Truncate/pad each text to **1500 characters**. Rationale: median text is ~123 words * ~5 chars/word + spaces = ~740 chars. 1500 captures the vast majority of texts without excessive padding. Experiment with [1000, 1500, 2000].

4. **No text normalization beyond URL replacement**: Preserve all case, punctuation, spacing, and formatting. These are stylistic signals.

5. **PyTorch Dataset**: Return `(char_ids_text1, char_ids_text2, label)` as tensors. Use `DataLoader` with `shuffle=True` for training, `batch_size=64`.

### 2.3 Model Architecture (Exact Specifications)

```
SHARED ENCODER (Siamese -- same weights for both texts):

Input: (batch, 1500) -- character indices

1. Character Embedding:
   - nn.Embedding(vocab_size=97, embedding_dim=32, padding_idx=0)
   - Output: (batch, 1500, 32)

2. Parallel Conv1D Layers (multi-width, capturing different n-gram scales):
   - Conv1D_3: nn.Conv1d(in=32, out=128, kernel_size=3, padding=1) + ReLU
   - Conv1D_5: nn.Conv1d(in=32, out=128, kernel_size=5, padding=2) + ReLU
   - Conv1D_7: nn.Conv1d(in=32, out=128, kernel_size=7, padding=3) + ReLU
   - Concatenate along channel dim: (batch, 384, 1500)
   - MaxPool1d(kernel_size=3, stride=3): (batch, 384, 500)
   - Dropout(0.2)

3. BiLSTM:
   - nn.LSTM(input_size=384, hidden_size=128, num_layers=1, 
             batch_first=True, bidirectional=True, dropout=0)
   - Output: (batch, 500, 256)  [128*2 for bidirectional]

4. Additive (Bahdanau) Attention:
   - W_a: nn.Linear(256, 128)
   - v_a: nn.Linear(128, 1, bias=False)
   - scores = v_a(tanh(W_a(lstm_output)))  -> (batch, 500, 1)
   - attention_weights = softmax(scores, dim=1)
   - attended = sum(attention_weights * lstm_output, dim=1)  -> (batch, 256)
   
5. Projection:
   - nn.Linear(256, 128)
   - ReLU
   - Dropout(0.3)
   - Output: document embedding of dimension 128

COMPARISON + CLASSIFIER:

6. Comparison Layer:
   - Given v1, v2 (each batch x 128):
   - diff = |v1 - v2|  (absolute difference)
   - prod = v1 * v2    (element-wise product)
   - combined = cat([v1, v2, diff, prod], dim=1)  -> (batch, 512)

7. MLP Classifier:
   - nn.Linear(512, 256) + ReLU + Dropout(0.4)
   - nn.Linear(256, 64) + ReLU + Dropout(0.3)
   - nn.Linear(64, 1) + Sigmoid
   - Output: P(same author)

ADVERSARIAL TOPIC HEAD (Gradient Reversal):

8. Topic Prediction (auxiliary):
   - GradientReversalLayer(lambda=0.1)  -- reverses gradients during backprop
   - Applied to the document embedding (v1 or v2, randomly selected per batch)
   - nn.Linear(128, 64) + ReLU + Dropout(0.3)
   - nn.Linear(64, num_topics) + Softmax
   - Loss: CrossEntropyLoss on topic pseudo-labels
```

**Domain/Topic labels for adversarial head**: Two options (try both, pick best):

- **Option 1 (Preferred): Corpus-type heuristic labels**: Classify each text into its source corpus using simple heuristics:
  - Email (Enron): presence of "Subject:", "From:", "Date:", email-header patterns
  - Blog: presence of "urlLink", blog-style markers
  - Movie review: presence of movie/actor names, review-style language, ratings
  - Unknown: texts not matching any heuristic
  - This provides ~3-4 clean domain labels that directly capture the content/domain confound.

- **Option 2: TF-IDF + K-means clustering**: Compute TF-IDF vectors (word-level, max_features=5000) on all training texts, then apply K-means with K=8-12 clusters. Use cluster assignments as pseudo-topic labels. This is MORE ROBUST than LDA on short texts (LDA fails on texts < 50 words because Dirichlet priors dominate, producing near-uniform topic distributions that provide no useful signal for adversarial training).

The GRL forces the encoder to produce embeddings that are NOT predictive of domain/topic, thus focusing on style.

### 2.4 Training Procedure

**Optimizer**: AdamW
- lr: 1e-3 (with cosine annealing schedule)
- weight_decay: 1e-4
- betas: (0.9, 0.999)

**Learning rate schedule**: CosineAnnealingWarmRestarts
- T_0: 5 epochs
- T_mult: 2
- eta_min: 1e-6

**Loss function**: Composite loss with three components:
- **Primary**: Binary Cross-Entropy for the verification decision
- **Contrastive**: Cosine embedding loss at the embedding level (NOT optional -- this is a central training objective):
  - Same-author pairs: minimize `1 - cos(v1, v2)`
  - Different-author pairs: maximize `cos(v1, v2)` with margin (margin=0.3)
  - Use `torch.nn.CosineEmbeddingLoss(margin=0.3)` with target +1 for same-author, -1 for different-author
- **Adversarial**: Cross-Entropy for domain prediction (auxiliary)
- Total loss = BCE_loss + 0.2 * contrastive_loss + 0.1 * domain_adversarial_loss
- The GRL lambda starts at 0.0 and linearly ramps to 0.1 over the first 5 epochs (gradual introduction prevents destabilizing early training)
- The contrastive loss weight is introduced at epoch 2 (after 1 epoch of BCE-only warmup)

**Batch size**: 64

**Epochs**: Max 50, with early stopping (patience=7 on dev macro_f1)

**Regularization**:
- Dropout as specified in architecture (0.2, 0.3, 0.4 at different layers)
- Weight decay 1e-4
- Gradient clipping: max_norm=5.0
- Early stopping on dev F1

**Stylistic Invariance Training (adversarial augmentation -- NOT optional)**:
- **Character perturbation as stylistic robustness training**: With probability 0.05 per character, replace with a random character from the same category (letter->letter, digit->digit). Only during training. This forces the model to learn stylistic representations robust to surface-level character variation -- acting as adversarial augmentation that prevents overfitting to exact character sequences while preserving higher-level stylistic patterns.
- **Random truncation**: Randomly truncate texts to 80-100% of their length during training. This teaches the model to extract style from variable-length inputs, simulating the real-world scenario where texts of different lengths must be compared.

**Checkpointing**: Save model state_dict at best dev macro_f1 epoch

**Hardware**: GPU (V100/A100 on Colab/university cluster). Training should take 1-3 hours depending on GPU.

### 2.5 Hyperparameter Sensitivity Analysis

Run ablation over:
- Sequence length: [1000, 1500, 2000]
- CNN filter count: [64, 128, 256]
- LSTM hidden size: [64, 128, 256]
- Dropout: [0.2, 0.3, 0.4, 0.5]
- Learning rate: [5e-4, 1e-3, 2e-3]
- With/without topic adversarial head
- Comparison method: [|v1-v2| only, [|v1-v2|, v1*v2], [v1, v2, |v1-v2|, v1*v2]]

### 2.6 Creativity Justification

**Central novel contributions (these define the creativity narrative):**

1. **Adversarial Style-Content Disentanglement (CENTRAL contribution)**: The GRL-based adversarial training is the defining innovation. While domain adaptation via GRL (Ganin & Lempitsky, 2015) is established in transfer learning, applying it specifically to DISENTANGLE style from content in AV is a novel contribution. The key insight: cross-domain AV datasets conflate topic similarity with authorship -- our adversarial head explicitly deconfounds this. Cite: Ganin & Lempitsky (2015) for GRL; arXiv:2411.18472 for the AV-specific style-content disentanglement motivation.

2. **Contrastive Embedding Learning for AV**: Cosine embedding loss at the representation level (not just classification loss) shapes the embedding space to have geometric structure: same-author pairs are close, different-author pairs are far. This is a primary training objective, not an auxiliary loss. This draws from metric learning (Bromley et al. 1993, original Siamese network) and contrastive NLP (Gao et al. 2021, SimCSE).

3. **Stylistic Invariance Training via Character Perturbation**: Random character noise is framed as adversarial augmentation for stylistic robustness -- the model must learn to extract style despite surface-level character variation. This is analogous to adversarial training in NLP robustness literature but applied to stylometric signal preservation.

4. **Interpretable Attention Weights (XAI contribution)**: The additive (Bahdanau) attention mechanism produces per-character-position weights that indicate which text regions the model considers most stylistically informative. These attention weights can be visualized to provide explainability -- showing, for example, that the model focuses on punctuation patterns and function word sequences rather than content words. This addresses the growing demand for XAI in NLP systems. Cite: Bahdanau et al. (2015).

**Foundational architecture (acknowledged prior work):**
- **Siamese character-level architecture**: Builds on Boenninghoff et al. (2019, 2021) -- the O2D2 system that won PAN 2020/2021. Our work extends this foundation with the adversarial disentanglement and contrastive objectives above.
- **Character-level CNN processing**: Cite Zhang, Zhao & LeCun (2015); Saedi & Dras (2021)
- **Multi-width CNN**: Draws from Kim (2014) adapted to character level

### 2.7 Soundness Checklist

- Siamese architecture uses shared weights (a single encoder, applied to both texts) -- confirmed correct for verification tasks
- Character-level input preserves stylistic signals (no tokenization artifacts)
- GRL lambda is ramped gradually to avoid destabilizing training
- The attention mechanism is additive (Bahdanau-style), NOT multi-head self-attention (which would be a transformer component and violate Category B constraints)
- No pre-trained transformer weights anywhere in the pipeline
- Conv1D padding chosen to preserve sequence length before pooling
- Gradient clipping prevents exploding gradients with LSTMs
- The comparison layer includes both difference AND product, capturing both distance and interaction patterns

---

## SOLUTION 3: CATEGORY C -- Style-Aware DeBERTa with Siamese Architecture, Contrastive Learning, and Layer-Weighted Style Representations

### 3.1 Architecture Overview

A Siamese DeBERTa-v3-base that encodes each text independently (not as a concatenated pair), producing style-focused representations by weighting earlier transformer layers more heavily (since layers 1-4 encode syntax/style while layers 8-12 encode semantics). A learnable layer-weighting mechanism (scalar mix) combines all 12 hidden states. The model is trained with a composite loss: cosine embedding contrastive loss (pulling same-author pairs together, pushing different-author pairs apart in embedding space) plus binary cross-entropy for the classification head. An optional gradient reversal topic head provides additional debiasing.

### 3.2 Data Preprocessing

1. **Tokenization**: Use DeBERTa-v3-base tokenizer (from HuggingFace `microsoft/deberta-v3-base`)
2. **Max sequence length**: 256 tokens per text (NOT 512 for the concatenated pair -- each text is encoded separately). Most texts are ~100 words which is approximately 130-150 subword tokens. 256 provides headroom.
3. **Input format**: Unlike standard cross-encoder (which concatenates text_1 [SEP] text_2), this is a Siamese encoder -- each text is tokenized and encoded independently: `[CLS] text_1 [SEP]` and `[CLS] text_2 [SEP]`
4. **Why Siamese, not cross-encoder**: Cross-encoder conflates content similarity with style similarity. By encoding texts separately, we force the model to learn per-text style representations that are then compared. This is a deliberate architectural choice that must be justified in the model card.

### 3.3 Model Architecture (Exact Specifications)

```
SHARED ENCODER (Siamese):

1. DeBERTa-v3-base Encoder:
   - Load from microsoft/deberta-v3-base (HuggingFace)
   - 12 layers, 768 hidden dim, 12 attention heads
   - output_hidden_states=True
   - Take [CLS] token representation from each of the 12 layers

2. Scalar Mix (Learnable Layer Weighting):
   - 12 learnable scalar weights (initialized uniformly at 1/12)
   - softmax over weights, then weighted sum of [CLS] from each layer
   - weighted_cls = sum(softmax(w_i) * hidden_state_i[:, 0, :])
   - This produces a single 768-dim vector per text
   - CRITICAL: Initialize with slight bias toward early layers 
     (e.g., w_1..w_4 = 0.12, w_5..w_12 = 0.07) to encourage 
     style-focused representations. The model can learn to adjust.

3. Style Projection Head:
   - nn.Linear(768, 256) + LayerNorm + GELU + Dropout(0.1)
   - nn.Linear(256, 128)
   - L2-normalize to unit sphere (for contrastive loss)
   - Output: 128-dim style embedding per text

COMPARISON + CLASSIFIER:

4. Comparison Layer:
   - Given v1, v2 (each batch x 128, L2-normalized):
   - diff = |v1 - v2|
   - prod = v1 * v2
   - cos_sim = (v1 * v2).sum(dim=1, keepdim=True)  -- scalar
   - combined = cat([v1, v2, diff, prod, cos_sim], dim=1)  -> (batch, 513)

5. Classification MLP:
   - nn.Linear(513, 256) + GELU + Dropout(0.2)
   - nn.Linear(256, 1) + Sigmoid
   - Output: P(same author)

AUXILIARY: Gradient Reversal Topic Head (same as Category B):
   - GRL(lambda=0.05) on style embedding
   - nn.Linear(128, 64) + ReLU + nn.Linear(64, 10)
```

### 3.4 Training Procedure

**Phase 1: Warm-up with BCE only (epochs 1-3)**
- Freeze DeBERTa layers 0-8, train layers 9-11 + all heads
- Optimizer: AdamW, lr=2e-5, weight_decay=0.01
- Loss: BCE only
- Purpose: stabilize the classification head before introducing contrastive loss

**Phase 2: Full training with composite loss (epochs 4-20+)**
- Unfreeze all layers
- Optimizer: AdamW with discriminative learning rates:
  - DeBERTa layers 0-3: lr=5e-6
  - DeBERTa layers 4-7: lr=1e-5
  - DeBERTa layers 8-11: lr=2e-5
  - Scalar mix weights: lr=1e-3
  - Projection head + MLP: lr=5e-4
- Loss: `L = BCE_loss + 0.3 * contrastive_loss + 0.05 * topic_adversarial_loss`
- **Contrastive Loss (Cosine Embedding Loss)**: The original plan used SupCon (Khosla et al. 2020), but SupCon assumes cross-pair author identity knowledge (i.e., knowing that text_1 from pair_A and text_1 from pair_B share the same author), which AV pair-level data does NOT provide. Instead, use **cosine embedding loss** which operates correctly at the pair level:
  - Same-author pairs: minimize `1 - cos(v1, v2)`
  - Different-author pairs: maximize `cos(v1, v2)` with margin
  - Implementation: `torch.nn.CosineEmbeddingLoss(margin=0.3)` with target +1 for same-author, -1 for different-author
  - This is simpler, mathematically correct for pair-level data, and provides the same embedding-space shaping benefits

**Learning rate schedule**: Linear warmup (first 10% of steps) then cosine decay to 0

**Batch size**: 32 (may need to reduce to 16 if GPU memory is insufficient with two separate forward passes per pair)

**Epochs**: Max 25, early stopping patience=5 on dev macro_f1

**Gradient accumulation**: 8 steps (effective batch size = 256). Larger effective batch size is critical for contrastive learning -- with small batches, the contrastive signal is too noisy to provide meaningful gradient updates. 256 is the minimum recommended effective batch size for stable contrastive training.

**Mixed precision**: fp16 training via torch.cuda.amp

### 3.5 Creativity Justification

- **Siamese transformer for AV**: Departure from standard cross-encoder. Cite Reimers & Gurevych (2019) "Sentence-BERT" for Siamese transformer architecture, adapted for style rather than semantic similarity
- **Learnable layer-weighted representations**: Cite Jawahar et al. (2019) "What Does BERT Learn about the Structure of Language?" which showed early layers encode syntax and later layers encode semantics. For style-based tasks, early layers are more relevant. The scalar mix approach is from Peters et al. (2018) ELMo.
- **Contrastive embedding learning**: Cosine embedding loss for pair-level contrastive learning (corrected from SupCon which requires cross-pair author identity). Cite Bromley et al. (1993) for original Siamese contrastive learning, Gao et al. (2021) "SimCSE" for contrastive NLP
- **Style-content disentanglement via GRL**: Same citations as Category B
- **Discriminative learning rates**: Cite Howard & Ruder (2018) "Universal Language Model Fine-tuning for Text Classification" (ULMFiT)
- **DeBERTa-v3**: Cite He et al. (2021) "DeBERTa: Decoding-enhanced BERT with Disentangled Attention" -- stronger than RoBERTa on most benchmarks

### 3.6 Soundness Checklist

- DeBERTa-v3-base is a pre-trained model available from HuggingFace, not external training data. Using a pre-trained language model is standard and allowed in closed mode (the restriction is on task-specific external datasets, not general pre-training).
- Siamese encoding is correct for verification: it prevents the model from learning spurious cross-text attention patterns that conflate content overlap with authorship
- Layer weighting is initialized but learnable -- the model determines the optimal balance
- Cosine embedding loss is the correct contrastive formulation for pair-level AV data (SupCon would require cross-pair author identity, which is unavailable)
- Discriminative LR is sound: lower LR for earlier (more general) layers prevents catastrophic forgetting
- Gradient accumulation compensates for small batch size
- L2 normalization of embeddings before contrastive loss is required for stable training

---

## EVALUATION PLAN (3 marks for Evaluation criterion)

This section targets full marks on the "Evaluation" criterion by going significantly beyond Codabench benchmarking.

**CRITICAL REQUIREMENT: Written Interpretation**: For EVERY plot, table, and statistical test in this evaluation plan, include 3-5 sentences of written analysis explaining: (a) what the result shows, (b) why it matters, and (c) what it implies for the model's strengths/weaknesses. Plots without interpretation score poorly on the rubric.

### 6.1 Quantitative Evaluation (for each of the two submitted solutions)

1. **Full metric suite on dev set**: Report all 8 metrics from the scorer (accuracy, macro precision, macro recall, macro_f1, weighted precision, weighted recall, weighted F1, MCC). Present in a table.

2. **Confusion matrices**: For each solution, plot a confusion matrix heatmap (using seaborn/matplotlib). Report TP, FP, TN, FN counts and rates.

3. **Per-class precision, recall, F1**: Breaking down performance on class 0 (different author) vs class 1 (same author). Identify whether the model has a bias toward one class.

4. **Calibration analysis**: Plot reliability diagrams (predicted probability vs observed frequency) for each model. Use `sklearn.calibration.calibration_curve`. This shows whether the model's probability outputs are well-calibrated.

5. **ROC curves and AUC**: Plot ROC curves for each solution. Report AUC. For Category A, this requires probability outputs from the ensemble.

6. **Precision-Recall curves**: More informative than ROC for imbalanced or near-balanced classification. Report area under PR curve.

### 6.2 Statistical Significance Testing

7. **McNemar's test**: Compare each solution against its category baseline (SVM for Cat A, LSTM for Cat B, BERT for Cat C). McNemar's test operates on the contingency table of correct/incorrect predictions between two classifiers. Report the chi-squared statistic and p-value. A p-value < 0.05 means the improvement is statistically significant, which the rubric explicitly requires.
   - **IMPORTANT**: McNemar's requires PAIRED predictions (both models' predictions on the same samples). The baseline predictions file at `/Users/kumar/Documents/University/Year3/NLU/project/baseline_extracted/nlu_bundle-feature-unified-local-scorer/baseline/25_DEV_AV.csv` should contain these. Verify that it has per-sample predictions (not just aggregate scores). If paired predictions are unavailable for a baseline, fall back to a **bootstrap difference test**: resample dev predictions 1000 times, compute macro_f1 for both models on each resample, and test whether the mean difference is significantly > 0.

8. **Bootstrap confidence intervals**: Compute 95% confidence intervals for macro_f1 and MCC via bootstrap resampling (1000 iterations) on the dev set. This quantifies uncertainty in the performance estimates.

9. **Paired bootstrap test between solutions**: If submitting A+B or A+C, compare the two solutions against each other using paired bootstrap to determine which is statistically better.

### 6.3 Error Analysis

10. **Misclassification analysis by text properties**:
    - Group dev pairs by combined text length (short <100 words, medium 100-200, long >200) and report F1 per group
    - Group by corpus type (Enron emails, blog posts, movie reviews) using the following concrete heuristics:
      - **Email (Enron)**: text contains "Subject:", "From:", "Date:", or email-header patterns (regex: `^(From|To|Subject|Date|Sent):`)
      - **Blog**: text contains "urlLink" tag
      - **Movie review**: text contains movie/actor proper nouns, ratings (e.g., "10/10", "stars"), or review-style phrases
      - **Unknown**: texts not matching any heuristic (report this count)
    - Report F1 per corpus-type group
    - Group by content overlap (high Jaccard similarity vs low) and report F1 per group. This directly tests whether the model is relying on topic overlap as a shortcut.

11. **False positive analysis**: Manually examine 20 false positives (predicted same-author but actually different). Characterize WHY the model failed: similar writing style? same topic? similar text length?

12. **False negative analysis**: Manually examine 20 false negatives (predicted different-author but actually same). Characterize WHY: different topics? different text lengths? unusual style variation?

### 6.4 Ablation Studies

13. **Feature ablation for Category A**: Train the ensemble with subsets of features removed:
    - All features (baseline)
    - Without character n-gram TF-IDF
    - Without function words
    - Without vocabulary richness
    - Without NCD features
    - Without pairwise similarity features
    - Only character n-gram TF-IDF + cosine similarity (minimal PAN 2022 baseline)
    - Report macro_f1 for each ablation

14. **Architecture ablation for Category B**:
    - Full model (CNN+BiLSTM+Attention+GRL+Contrastive)
    - Without attention (global max pool instead)
    - Without BiLSTM (CNN only, global max pool)
    - Without domain adversarial head (GRL)
    - Without contrastive loss (BCE only)
    - Without stylistic invariance training (no character perturbation)
    - Different comparison methods: |v1-v2| only vs full [v1, v2, |v1-v2|, v1*v2]
    - Report macro_f1 for each

15. **Attention visualization for Category B (XAI analysis)**:
    - For 10 correctly classified same-author pairs and 10 correctly classified different-author pairs, extract and visualize the attention weights over the character sequence
    - Show which text regions (punctuation clusters, function words, formatting) receive highest attention
    - Compare attention patterns between same-author and different-author pairs
    - This provides interpretability evidence supporting the model's use of stylistic (not topical) features

16. **Layer weighting analysis for Category C** (if submitted):
    - Visualize the learned scalar mix weights to show which layers the model relies on
    - Compare with uniform weighting and last-layer-only

### 6.5 Cross-Solution Analysis

17. **Agreement analysis**: Compute Cohen's kappa between solutions. Analyze the pairs where the two solutions disagree. Are there systematic differences in what each approach handles well?

18. **Ensemble of solutions**: What happens if we majority-vote or average probabilities across solutions? (Not for submission, but demonstrates analytical thinking.)

19. **Failure mode comparison (Cat A vs Cat B)**: Systematic analysis of WHY the two solutions disagree on specific pairs:
    - Extract the set of pairs where Cat A is correct but Cat B is wrong, and vice versa
    - For each disagreement set, compute summary statistics: average text length, corpus type distribution, content overlap (Jaccard), vocabulary richness
    - Identify systematic patterns: e.g., "Cat A struggles with short texts where vocabulary richness features are unreliable" or "Cat B struggles with formal email text where character-level patterns are less discriminative"
    - This demonstrates deep understanding of each approach's strengths and weaknesses

---

## MODEL CARD PLAN (13 marks: 3 formatting + 6 informativeness + 4 accuracy)

Each model card must use the template at `/Users/kumar/Documents/University/Year3/NLU/project/archive_extracted/COMP34812_modelcard_template.md` and be generated via the Jinja-based notebook at `/Users/kumar/Documents/University/Year3/NLU/project/archive_extracted/Model Card Creation.ipynb`.

### 7.1 Template Fields (For Each Model Card)

The template has these fields from the `ModelCard.from_template()` call: `model_id`, `model_summary`, `model_description`, `developers`, `base_model_repo`, `base_model_paper`, `model_type`, `model_architecture`, `language`, `base_model`, `training_data`, `hyperparameters`, `speeds_sizes_times`, `testing_data`, `testing_metrics`, `results`, `hardware_requirements`, `software`, `bias_risks_limitations`, `additional_information`.

### 7.2 Model Card for Category A (Diff-Vector Stacking Ensemble)

- **model_id**: `username1-username2-AV-CatA`
- **model_summary**: "A traditional machine learning system for authorship verification that uses comprehensive stylometric features (lexical, character, syntactic, structural) computed as diff-vectors between text pairs, classified by a stacking ensemble of SVM, Random Forest, and XGBoost with a logistic regression meta-learner."
- **model_description**: Detailed 2-3 paragraph description covering: the AV task definition, the diff-vector approach (cite Stamatatos et al. 2023), the feature groups (with counts), the ensemble architecture, and the key design rationale (style vs content).
- **developers**: Full names
- **base_model_repo**: "N/A (no pre-trained model used)"
- **base_model_paper**: List all key papers: Stamatatos (2009), Abbasi & Chen (2008), Stamatatos et al. (2023), Jiang et al. (2023), Weerasinghe & Greenstadt (2020)
- **model_type**: "Supervised classification"
- **model_architecture**: "Stacking ensemble (SVM-RBF + Random Forest + XGBoost, meta-learner: Logistic Regression) on stylometric diff-vectors"
- **language**: "English"
- **base_model**: "N/A"
- **training_data**: "27,643 pairs of texts from the COMP34812 AV training set, comprising Enron emails (~27%), blog posts (~15%), and movie reviews (~16%). Texts average ~100-123 words. Dataset is balanced: 50.5% same-author, 49.5% different-author."
- **hyperparameters**: List EXACT final hyperparameters for each base classifier and meta-learner. Include feature extraction parameters (n-gram ranges, TF-IDF settings, SVD components).
- **speeds_sizes_times**: Feature extraction time, training time per classifier, total training time, model size on disk
- **testing_data**: "5,993 pairs from the COMP34812 AV development set, used for evaluation. Performance was also assessed via 5-fold stratified cross-validation on the training set."
- **testing_metrics**: "Macro F1-score (primary), Matthews Correlation Coefficient (MCC), accuracy, macro precision, macro recall, weighted F1"
- **results**: Complete table of all metrics on dev set. Include comparison to SVM baseline (macro_f1=0.5610, MCC=0.1235). Include McNemar's p-value. Include key ablation results.
- **hardware_requirements**: "CPU only. RAM: 8GB minimum. No GPU required."
- **software**: Exact versions of scikit-learn, xgboost, spacy, numpy, pandas, gzip (stdlib)
- **bias_risks_limitations**: (1) Feature extraction assumes English text -- function word list and POS tagger are English-specific. (2) Vocabulary richness metrics are unreliable for very short texts (<30 words). (3) The model may learn corpus-specific biases (e.g., email vs blog formatting differences). (4) Character n-gram TF-IDF is fitted on training texts only; unseen character patterns in test data will have zero weight. (5) The model does not account for deliberate style obfuscation.
- **additional_information**: Feature importance ranking from Random Forest/XGBoost. Description of ablation study findings.

### 7.3 Model Card for Category B (Adversarial Style-Content Disentanglement Network)

Follow the same template with architecture-specific details:
- **model_summary**: "An Adversarial Style-Content Disentanglement Network for authorship verification. Uses gradient reversal to learn style representations explicitly disentangled from topic/content, with contrastive embedding learning and stylistic invariance training via character perturbation."
- **model_architecture**: "Siamese network: Character Embedding (32d) -> Multi-width Conv1D (3,5,7 kernels, 128 filters each) -> BiLSTM (128 hidden, bidirectional) -> Additive Attention -> 128d document embedding -> [v1, v2, |v1-v2|, v1*v2] comparison -> MLP classifier. Gradient reversal domain prediction head for adversarial style-content disentanglement. Cosine embedding contrastive loss at the embedding level."
- **base_model**: "N/A (trained from scratch)"
- **hyperparameters**: EXACT values: embedding_dim=32, conv_filters=128, lstm_hidden=128, dropout rates, optimizer params, lr schedule params, GRL lambda schedule, contrastive loss margin, batch size, epochs trained, early stopping epoch
- **Key framing**: Emphasize (1) adversarial disentanglement as the central contribution, (2) contrastive learning as a primary objective, (3) attention weights as interpretability/XAI contribution
- All other fields filled with equal specificity

### 7.4 Model Card for Category C (if submitted)

- **base_model**: "microsoft/deberta-v3-base"
- **base_model_repo**: "https://huggingface.co/microsoft/deberta-v3-base"
- **base_model_paper**: "https://arxiv.org/abs/2111.09543"
- Detail the Siamese architecture, scalar mix, contrastive learning, discriminative LRs

### 7.5 Accuracy of Model Cards (4 marks)

The highest-weighted single criterion. Every claim in the model card must be verifiable from the code. Specific measures:
- Hyperparameters in the card must match the code EXACTLY (grep through the final training notebook to verify)
- Architecture descriptions must match the model class definitions
- Training procedure must match the training loop
- Results must match the scorer output
- Do a final review pass: read each sentence of each model card and check it against the code

### 7.6 Model Card Verification Protocol (NEW -- concrete checklist)

This is a formal verification protocol to be executed by BOTH team members independently before submission. Each check must be signed off.

**Hyperparameter Verification:**
- [ ] For each hyperparameter listed in the model card: `grep` the training code/notebook for the exact value. If it does not match, update the model card (NOT the code).
- [ ] Verify: learning rate, batch size, epochs trained (actual, not max), optimizer, weight decay, dropout rates, hidden dimensions, number of layers, regularization parameters.

**Architecture Verification:**
- [ ] For each architectural claim in the model card: compare against `print(model)` output or the model class definition in code.
- [ ] Verify: layer types, layer dimensions, activation functions, number of parameters, comparison method.

**Metric Verification:**
- [ ] For each metric reported in the model card: compare against the EXACT output of running the scorer (`nlu-score --task av --prediction <path>`).
- [ ] Cross-check: copy-paste the scorer output line into the model card. Do not round or paraphrase.

**Data Verification:**
- [ ] For each data claim (dataset size, class distribution, text statistics): compare against `len(df)`, `df['label'].value_counts()`, and descriptive statistics computed from the actual data.
- [ ] Verify: number of training pairs, number of dev pairs, class balance percentages, text length statistics.

**Software Version Verification:**
- [ ] For each library listed: verify the version matches what is actually installed in the training environment (`pip show <package>`).

**Cross-check Protocol:**
- [ ] Team member 1 fills in the model card.
- [ ] Team member 2 independently runs through this checklist, marking each item as verified or flagging discrepancies.
- [ ] All flagged discrepancies are resolved before submission.

### 7.7 spaCy Closed-Mode Compliance Note

spaCy (`en_core_web_md`) is a general-purpose pre-trained NLP pipeline (POS tagging, dependency parsing, NER). It is NOT external AV-specific data or an AV-specific model. It is comparable to using NLTK or Stanford CoreNLP for linguistic annotation. This should be explicitly noted in the model card under "Software" or "Additional Information" to preempt any closed-mode compliance questions. The model's word vectors are used only for POS/dependency accuracy, not as features themselves.

---

## CODE ORGANIZATION PLAN (3 marks Organisation + 3 marks Completeness)

### 8.1 Directory Structure

```
submission/
├── README.md                          # Project overview, how to run, attributions, AI tool declaration
├── requirements.txt                   # All Python dependencies with versions
├── Group_n_A.csv                      # Category A predictions on test set
├── Group_n_B.csv                      # Category B predictions on test set  (or Group_n_C.csv)
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb    # Data exploration, statistics, visualizations
│   ├── 02_CatA_Training.ipynb            # Category A: feature engineering + ensemble training
│   ├── 03_CatB_Training.ipynb            # Category B: Siamese network training
│   ├── 04_CatC_Training.ipynb            # Category C: DeBERTa training (if submitted)
│   ├── 05_Evaluation.ipynb               # All evaluation: confusion matrices, McNemar's, ablation, error analysis
│   ├── 06_Demo_CatA.ipynb                # Demo: load model, run inference on input CSV, produce predictions
│   ├── 07_Demo_CatB.ipynb                # Demo: load model, run inference
│   └── 08_Model_Card_Generation.ipynb    # Generate model cards from template
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                     # Data loading, preprocessing, character encoding
│   ├── feature_engineering.py            # All stylometric feature extraction for Cat A
│   ├── models/
│   │   ├── __init__.py
│   │   ├── siamese_cnn_bilstm.py        # Cat B: PyTorch model definition
│   │   ├── style_deberta.py             # Cat C: DeBERTa Siamese model definition
│   │   └── gradient_reversal.py         # GRL layer implementation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_ensemble.py            # Cat A training logic
│   │   ├── train_siamese.py             # Cat B training loop
│   │   └── train_deberta.py             # Cat C training loop
│   └── evaluation/
│       ├── __init__.py
│       └── eval_utils.py                # Confusion matrix, McNemar's, bootstrap CI, error analysis
│
├── models/                               # Saved trained models (or link to cloud storage if >10MB)
│   ├── cat_a_ensemble.joblib             # Category A: fitted sklearn pipeline
│   ├── cat_a_tfidf.joblib                # Category A: fitted TF-IDF vectorizer
│   ├── cat_a_svd.joblib                  # Category A: fitted SVD transformer
│   ├── cat_a_scaler.joblib               # Category A: fitted StandardScaler
│   ├── cat_b_siamese.pt                  # Category B: PyTorch state_dict
│   └── cat_c_deberta.pt                  # Category C: PyTorch state_dict (or HF save format)
│
├── model_cards/
│   ├── model_card_catA.md                # Generated model card for Category A
│   └── model_card_catB.md                # Generated model card for Category B (or C)
│
└── poster/
    └── poster.pdf                        # A1 landscape or 16:9 PowerPoint PDF
```

### 8.2 README.md Contents

1. **Project overview**: Track, task, team members
2. **Solutions summary**: One paragraph per solution
3. **How to install**: `pip install -r requirements.txt` or `conda env create -f environment.yml`
4. **How to train**: Point to each training notebook with clear instructions
5. **How to run demo/inference**: Step-by-step instructions for each demo notebook. "Given a CSV file with columns text_1 and text_2, the demo code loads the trained model and produces a predictions CSV."
6. **How to evaluate**: Point to evaluation notebook
7. **Code structure**: Directory tree with descriptions
8. **Model storage**: If models >10MB, provide OneDrive/cloud link
9. **Data attribution**: "Training and evaluation data provided by the COMP34812 teaching team"
10. **Code attribution**: Any libraries or code snippets reused (with links)
11. **Use of Generative AI Tools**: Declaration as required by Section VI of coursework spec

### 8.3 Documentation Standards

- Every notebook has markdown cells explaining what each code block does
- Every function has a docstring with parameters, return type, and brief description
- Every hyperparameter is defined as a named constant at the top of the notebook/file (not magic numbers)
- Inline comments for non-obvious logic
- Type hints in function signatures

### 8.4 Completeness Checklist

- All trained models saved and loadable
- Demo notebooks: given a CSV path, produce predictions end-to-end
- **CRITICAL: All demo notebooks MUST start with `!pip install` commands** for all required packages (e.g., `!pip install scikit-learn xgboost spacy torch pandas numpy`). Do not assume the grader has any packages pre-installed beyond base Python.
- No hardcoded file paths (use relative paths or configurable paths)
- `requirements.txt` includes ALL dependencies with pinned versions
- All random seeds set for reproducibility (numpy, torch, sklearn, python random)
- Demo code includes `!pip install` cells for any non-standard packages
- For spaCy: include `!python -m spacy download en_core_web_md` in the setup cell

### 8.5 Poster Content Plan

The poster (A1 landscape or 16:9 PowerPoint PDF) must include all of the following, per the coursework spec:

1. **Task and Track**: Brief description of the AV task and that we are on the AV track
2. **Dataset Summary**: Number of training/dev pairs, class balance, corpus types (Enron emails, blog posts, movie reviews), text length statistics
3. **Both Methods**: Clear description of each submitted solution (e.g., Cat A: Stylometric Diff-Vector Ensemble; Cat B: Adversarial Style-Content Disentanglement Network). Include a small architecture diagram for each.
4. **Dev Results Table**: Macro F1, MCC, and accuracy for both solutions, alongside the baselines. Highlight the gap over baselines.
5. **Brief Error Analysis**: 2-3 key findings from the error analysis (e.g., "Cat A struggles with short texts", "Cat B is robust to cross-topic pairs")
6. **Limitations and Ethical Considerations**: Acknowledge limitations (closed-mode constraints, English-only, vulnerability to style mimicry) and ethical issues (authorship verification can be used for surveillance, de-anonymization)

---

## DECISION FRAMEWORK: SELECTING THE BEST TWO SOLUTIONS

### 9.1 Decision Criteria (in priority order)

1. **Dev set macro_f1 gap over baseline**: The primary metric. Compute the gap between each solution and its category baseline:
   - Cat A gap = solution_A_f1 - 0.5610
   - Cat B gap = solution_B_f1 - 0.6226
   - Cat C gap = solution_C_f1 - 0.7854
   
2. **Statistical significance**: Run McNemar's test. If a solution's improvement over its baseline is NOT statistically significant (p > 0.05), it risks getting 0/3 on competitive performance for that solution.

3. **MCC**: As a secondary metric. The baselines have low MCC (0.12, 0.25, 0.57). A high MCC improvement is impressive.

4. **Creativity headroom**: A solution with higher architectural creativity scores better on the 3-mark creativity criterion, even if its absolute F1 is slightly lower.

### 9.2 Expected Decision Matrix

| Metric | Cat A Expected | Cat B Expected | Cat C Expected |
|--------|---------------|----------------|----------------|
| Dev macro_f1 | 0.65-0.75 | 0.70-0.78 | 0.78-0.84 |
| Baseline macro_f1 | 0.5610 | 0.6226 | 0.7854 |
| Expected gap | +0.09-0.19 | +0.08-0.16 | -0.01-0.06 |
| McNemar significant? | Very likely | Very likely | Possible |
| Creativity score | Very high | Very high | High |

**Note on revised expectations**: The original ranges (Cat A: 0.72-0.78, Cat B: 0.75-0.82, Cat C: 0.80-0.86) were optimistic. Cross-domain AV is genuinely hard -- the dataset mixes Enron emails, blog posts, and movie reviews, creating strong topic confounds. The revised ranges are more realistic and account for the difficulty of disentangling style from content.

### 9.3 Decision Rules

- **If Cat A > 0.65 AND Cat B > 0.70**: Submit A + B. Both have meaningful gaps over baselines, both are highly creative, and this combination is the most distinctive.
- **If Cat C > Cat B by more than 0.05 F1 AND Cat C > 0.80**: Consider submitting A + C instead, since the absolute performance advantage of C may matter for the competitive performance marks.
- **If Cat B fails to beat 0.65 (close to LSTM baseline)**: Discard B, submit A + C.
- **If Cat A fails to beat 0.58**: This would be surprising, but discard A and submit B + C.
- **Default recommendation**: A + B (highest combined creativity, largest expected gaps over baselines, most distinctive combination).

---

## PREDICTION GENERATION WORKFLOW (Test Data March 24)

### 10.1 Pre-Test-Data Preparation (Before March 24)

1. **All three models fully trained and validated** on dev set
2. **Prediction pipelines tested**: For each model, write and test a `predict(csv_path) -> predictions` function using the dev CSV as a dry run. Verify that the output matches the expected format.
3. **Output format verified**: Single column, integers (0 or 1), 5993 rows for dev. For test, the number of rows will match the test CSV. No header needed (but the scorer can handle a non-numeric header line).
4. **Scorer tested**: Run `nlu-score --task av --prediction path/to/dev_predictions.csv` and verify the scores match expectations.

### 10.2 Test Day Workflow (March 24)

1. **Download test CSV** from Canvas. Expected format: `text_1,text_2` (no label column).
2. **Verify format**: Check column names, row count, no label column.
3. **Run Category A prediction**:
   - Load saved TF-IDF vectorizer, SVD, scaler, ensemble model
   - Extract features from test pairs (same pipeline as training)
   - Generate predictions
   - Save as `Group_n_A.csv`
4. **Run Category B prediction**:
   - Load saved PyTorch model
   - Character-encode test texts
   - Run forward pass with `model.eval()` and `torch.no_grad()`
   - Apply threshold (0.5) to sigmoid output
   - Save as `Group_n_B.csv`
5. **Run Category C prediction** (if applicable):
   - Same as B but with DeBERTa model
   - Save as `Group_n_C.csv`
6. **Verify predictions**: Check row count matches test set, values are 0 or 1, no NaN/null
7. **Select best two** based on the decision framework above (using dev performance as proxy)
8. **Final sanity check**: Ensure prediction file names follow convention `Group_n_A.csv` / `Group_n_B.csv`

### 10.3 Submission Assembly (Before March 31 14:00)

1. Package into a single zip file:
   - Prediction CSVs
   - All notebooks (training + evaluation + demo)
   - Source code modules
   - Trained models (or cloud links if >10MB)
   - Model cards (2 markdown files)
   - Poster PDF
   - README.md
2. Upload to Canvas

---

## RISK MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Cat A feature engineering takes too long | Low | Medium | Start with char n-gram TF-IDF + cosine similarity (the minimal PAN 2022 approach that works). Add features incrementally. |
| Cat B Siamese network fails to converge | Medium | High | Have a simpler fallback: pure CNN Siamese (no BiLSTM, no attention). Start with BCE loss, not contrastive. |
| Cat C DeBERTa does not beat BERT baseline significantly | Medium | High | This is why we build all 3: if Cat C only marginally beats BERT, submit A+B instead. |
| Topic confound inflates dev scores but fails on test | Medium | High | The GRL topic debiasing addresses this for B and C. For Cat A, NCD and character n-gram features are topic-resistant. Also run the content overlap error analysis to detect the problem early. |
| GPU training takes too long | Low | Medium | Cat B should train in 1-3 hours. Cat C in 2-4 hours. Use Colab Pro or university cluster. |
| Model files exceed 10MB | Likely for B,C | Low | Upload to OneDrive, include link in README as instructed in coursework spec. |
| spaCy POS tagger is slow on 55K+ texts | Medium | Low | Batch processing with `nlp.pipe()`, disable unnecessary components (`ner` -- keep `parser` for dependency features). Use `en_core_web_md` for accuracy. |

---

## TIMELINE

| Day | Tasks |
|-----|-------|
| Day 1 (Mar 22) | Set up project structure. Implement data loading and preprocessing. Begin Cat A feature engineering (Groups 1-4: lexical, character, char n-gram TF-IDF, function words). |
| Day 2 (Mar 23) | Complete Cat A features (Groups 5-9: POS, structural, syntactic complexity, writing rhythm, info-theoretic + topic-robustness mechanism + pairwise features). Train initial Cat A ensemble. Begin Cat B data pipeline (character encoding, Dataset, DataLoader). |
| Day 3 (Mar 24) | **Test data released.** Finish Cat B model implementation (including contrastive loss, domain labels, stylistic invariance training). Begin Cat B training. Begin Cat C model implementation. Generate Cat A test predictions immediately. |
| Day 4 (Mar 25) | Complete Cat B training and tuning. Complete Cat C training. Generate all test predictions. Run full evaluation suite (confusion matrices, McNemar's, ablation). |
| Day 5 (Mar 26) | Error analysis (including failure mode comparison between Cat A and Cat B). Select best two solutions. Begin model cards. |
| Day 6 (Mar 27) | Complete model cards. Run Model Card Verification Protocol (Section 7.6). Polish code and documentation. |
| Day 7 (Mar 28) | Demo notebooks with `!pip install` commands. Attention visualization for Cat B. Written interpretation for all plots/tables. |
| Day 8 (Mar 29) | Poster content: task/track, dataset summary, both methods, dev results table, error analysis, limitations/ethical considerations. |
| Day 9 (Mar 30) | Final review pass. Both team members independently verify model cards. Package submission zip. |
| Day 10 (Mar 31, before 14:00) | Last-minute fixes. Upload to Canvas. |

---

## PAPERS TO CITE (Complete Reference List)

### Category A Citations
1. Stamatatos, E. (2009). "A survey of modern authorship attribution methods." JASIST, 60(3).
2. Stamatatos, E., et al. (2023). "Same or Different? Diff-Vectors for Authorship Analysis." ACM TKDD.
3. Abbasi, A. & Chen, H. (2008). "Writeprints: A Stylometric Approach to Identity-level Identification and Similarity Detection." ACM TOIS.
4. Jiang, Z., et al. (2023). "Low-Resource Text Classification: A Parameter-Free Classification Method with Compressors." ACL Findings.
5. Weerasinghe, J. & Greenstadt, R. (2020). "Feature Vector Difference based Neural Network and Logistic Regression Models for Authorship Verification." PAN@CLEF 2020.
6. Bevendorff, J., et al. (2022). "Overview of the Authorship Verification Task at PAN 2022." CLEF 2022.
7. Burrows, J. (2002). "'Delta': A Measure of Stylistic Difference and a Guide to Likely Authorship." Literary and Linguistic Computing.
8. Wolpert, D. (1992). "Stacked Generalization." Neural Networks.

### Category A Additional Citations (Novel Features)
9. Feng, S., Banerjee, R., & Choi, Y. (2012). "Characterizing Stylistic Elements in Syntactic Structure." EMNLP. (Syntactic complexity for stylometry)
10. Shannon, C. E. (1948). "A Mathematical Theory of Communication." Bell System Technical Journal. (Information-theoretic foundations)

### Category B Citations
11. Boenninghoff, B., et al. (2019). "Explainable Authorship Verification in Social Media via Attention-based Similarity Learning." arXiv:1910.08144.
12. Boenninghoff, B., et al. (2021). "O2D2: Out-of-Distribution Detector to Capture Undecidable Trials." arXiv:2106.15825.
13. Zhang, X., Zhao, J., & LeCun, Y. (2015). "Character-level Convolutional Networks for Text Classification." NeurIPS.
14. Saedi, C. & Dras, M. (2021). "Siamese Networks for Large-Scale Author Identification." CSUR.
15. Bahdanau, D., Cho, K., & Bengio, Y. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR.
16. Ganin, Y. & Lempitsky, V. (2015). "Unsupervised Domain Adaptation by Backpropagation." ICML.
17. Kim, Y. (2014). "Convolutional Neural Networks for Sentence Classification." EMNLP.
18. Bromley, J., et al. (1993). "Signature Verification Using a Siamese Time Delay Neural Network." NIPS. (Original Siamese network / contrastive learning)

### Category C Citations
19. He, P., et al. (2021). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention." ICLR.
20. Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP.
21. Jawahar, G., Sagot, B., & Seddah, D. (2019). "What Does BERT Learn about the Structure of Language?" ACL.
22. Peters, M., et al. (2018). "Deep Contextualized Word Representations." NAACL (ELMo).
23. Khosla, P., et al. (2020). "Supervised Contrastive Learning." NeurIPS. (Referenced for context, but we use cosine embedding loss instead due to pair-level data constraints.)
24. Gao, T., Yao, X., & Chen, D. (2021). "SimCSE: Simple Contrastive Learning of Sentence Embeddings." EMNLP.
25. Howard, J. & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." ACL (ULMFiT).

### Style-Content Disentanglement
26. "Isolating Authorship from Content with Semantic Embeddings and Contrastive Learning." (2024). arXiv:2411.18472.

### PAN Competition Overviews
27. Kestemont, M., et al. (2020). "Overview of the Cross-Domain Authorship Verification Task at PAN 2020." CLEF 2020.
28. Kestemont, M., et al. (2021). "Overview of the Authorship Verification Task at PAN 2021." CLEF 2021.

---

### Critical Files for Implementation
- `/Users/kumar/Documents/University/Year3/NLU/project/training_extracted/training_data/AV/train.csv` - Primary training data: 27,643 text pairs with labels, all feature engineering and model training starts here
- `/Users/kumar/Documents/University/Year3/NLU/project/training_extracted/training_data/AV/dev.csv` - Development evaluation data: 6,004 rows (5,993 evaluated), used for all model selection and evaluation
- `/Users/kumar/Documents/University/Year3/NLU/project/baseline_extracted/nlu_bundle-feature-unified-local-scorer/local_scorer/metrics.py` - Scorer implementation: defines the exact metric functions (macro_f1, MCC, etc.) that will determine the grade; must match when validating predictions locally
- `/Users/kumar/Documents/University/Year3/NLU/project/archive_extracted/COMP34812_modelcard_template.md` - Model card Jinja template: the exact template that must be filled for the 13-mark model card section
- `/Users/kumar/Documents/University/Year3/NLU/project/baseline_extracted/nlu_bundle-feature-unified-local-scorer/baseline/25_DEV_AV.csv` - Baseline predictions file: contains the SVM, LSTM, and BERT baseline predictions on dev set, needed for McNemar's statistical significance tests comparing your solutions against baselines