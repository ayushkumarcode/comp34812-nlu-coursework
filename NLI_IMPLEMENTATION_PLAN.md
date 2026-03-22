# EXHAUSTIVE IMPLEMENTATION PLAN: COMP34812 Natural Language Inference -- Targeting 95-100%

## STRATEGIC OVERVIEW

**Track:** Natural Language Inference (NLI)
**Task:** Given a premise and a hypothesis, determine whether the hypothesis is entailed by (label=1) or contradicts/is neutral to (label=0) the premise. Binary classification.
**Strategy:** Build ALL THREE category solutions (A, B, C), then select the best two for submission. This maximizes optionality and de-risks the project.

**Submission constraints verified from the coursework spec and scorer:**
- Prediction files: `Group_n_A.csv`, `Group_n_B.csv`, or `Group_n_C.csv` -- single column of integers (0 or 1), no header needed (the scorer auto-skips a non-numeric header if present), one value per line
- Dev set: 6,735 evaluation rows (from `NLU_SharedTask_NLI_dev.solution`); dev.csv has 6,736 data rows (1 header + 6,736 data), train.csv has 24,432 data rows
- Labels: binary (0 = not entailed / contradiction / neutral, 1 = entailed). Train distribution: 48.2% label 0 (11,784), 51.8% label 1 (12,648). Dev distribution: 48.4% label 0 (3,258), 51.6% label 1 (3,478). Near-balanced.
- Premise word count: min=0, max=281, avg=18.9 words. Hypothesis word count: min=1, max=45, avg=10.4 words.
- Metrics: accuracy, macro precision, macro recall, **macro_f1** (primary), weighted precision, weighted recall, weighted F1, **matthews_corrcoef** (MCC) -- all computed by the local scorer at `/Users/kumar/Documents/University/Year3/NLU/project/baseline_extracted/nlu_bundle-feature-unified-local-scorer/`
- Baselines to beat: SVM macro_f1=0.5846; LSTM macro_f1=0.6603; BERT macro_f1=0.8198
- **Closed mode**: only provided training data allowed, no external datasets. Pre-trained models (GloVe, BERT, DeBERTa) are allowed as they are general-purpose, not task-specific.

**Key NLI-specific challenges:**
1. The task requires understanding semantic relationships between premise and hypothesis -- overlap features alone are insufficient
2. Negation and contradiction detection is critical (e.g., "He was fired" vs "He is hired")
3. Hypothesis-only bias: models may learn shortcuts from hypothesis-only cues without attending to premises (McCoy et al. 2019)
4. Lexical overlap does not always imply entailment (e.g., "A dog bites a man" vs "A man bites a dog")
5. Entailment often requires world knowledge and commonsense reasoning

---

## SOLUTION 1: CATEGORY A -- Feature-Rich Stacking Ensemble with Alignment and Knowledge Features

### 1.1 Architecture Overview

Extract ~280 features per premise-hypothesis pair spanning lexical overlap, semantic similarity, negation/contradiction signals, syntactic structure, and word alignment coverage. Feed into a stacking ensemble: base classifiers (XGBoost, LightGBM, SVM-RBF, Logistic Regression) whose out-of-fold predictions are meta-learned by a Logistic Regression classifier. The key creative elements are (a) monolingual word alignment features inspired by Sultan et al. (2014) and (b) natural logic relation features inspired by MacCartney & Manning (2007/2009).

### 1.2 Data Preprocessing

1. **Load CSV** via pandas with proper quoting handling (`quotechar='"'`, `escapechar=None`, use Python csv engine). The training data at `/Users/kumar/Documents/University/Year3/NLU/project/training_extracted/training_data/NLI/train.csv` has columns: `premise`, `hypothesis`, `label`.

2. **Text cleaning** (preserve semantic content while normalizing noise):
   - Decode HTML entities (`&amp;` -> `&`, etc.)
   - Normalize Unicode (NFC normalization)
   - Strip leading/trailing whitespace
   - Lowercase for feature extraction (but keep original case for some features)
   - Replace URLs with `<URL>` token
   - Normalize multiple spaces to single space
   - Handle empty premise edge case (min word count = 0 in training data): if premise is empty or whitespace-only, set to a placeholder `"."` to avoid division-by-zero in ratio features

3. **Tokenization**: Use spaCy `en_core_web_sm` for:
   - Word tokenization
   - POS tagging
   - Dependency parsing
   - Lemmatization
   - Named entity recognition (for entity overlap features)

4. **Label handling**: Labels in train.csv are integers (0, 1). No conversion needed.

5. **Train/dev split**: Use the provided split. For internal validation during development, use 5-fold stratified CV on training data.

### 1.3 Feature Engineering (~280 Features)

**Tier 1: Lexical Overlap Features (28 features)**

1. Unigram overlap ratio (premise -> hypothesis): |P_words ∩ H_words| / |H_words| -- what fraction of hypothesis words appear in premise
2. Unigram overlap ratio (hypothesis -> premise): |P_words ∩ H_words| / |P_words|
3. Bigram overlap ratio (premise -> hypothesis): |P_bigrams ∩ H_bigrams| / |H_bigrams|
4. Bigram overlap ratio (hypothesis -> premise): |P_bigrams ∩ H_bigrams| / |P_bigrams|
5. Trigram overlap ratio (premise -> hypothesis): |P_trigrams ∩ H_trigrams| / |H_trigrams|
6. Trigram overlap ratio (hypothesis -> premise): |P_trigrams ∩ H_trigrams| / |P_trigrams|
7. Jaccard similarity (unigram): |P ∩ H| / |P ∪ H|
8. Jaccard similarity (bigram): |P_bi ∩ H_bi| / |P_bi ∪ H_bi|
9. BLEU-1 (hypothesis as reference, premise as candidate) via nltk.translate.bleu_score
10. BLEU-2
11. BLEU-3
12. BLEU-4
13. BLEU-1 (premise as reference, hypothesis as candidate)
14. BLEU-2 (reverse)
15. BLEU-3 (reverse)
16. BLEU-4 (reverse)
17. Length ratio: len(premise_words) / len(hypothesis_words) (capped at 10 for stability)
18. Length difference: |len(premise_words) - len(hypothesis_words)|
19. Hypothesis coverage: fraction of hypothesis content words (non-stopword) found in premise
20. Premise coverage: fraction of premise content words found in hypothesis
21. Lemma overlap ratio (premise -> hypothesis): using spaCy lemmas
22. Lemma overlap ratio (hypothesis -> premise)
23. Lemma Jaccard similarity
24. Exact match: 1 if premise.strip().lower() == hypothesis.strip().lower() else 0
25. Substring containment: 1 if hypothesis.lower() in premise.lower() else 0
26. Character overlap ratio (character-level Jaccard on character trigrams)
27. Longest common subsequence ratio: LCS_length / max(len(P), len(H))
28. Longest common substring ratio: LCS_substring_length / max(len(P), len(H))

**Tier 2: Semantic Similarity Features (18 features)**

29. TF-IDF cosine similarity: Fit TF-IDF vectorizer (`max_features=20000`, `ngram_range=(1,2)`, `sublinear_tf=True`) on all training premises + hypotheses. For each pair, compute cosine similarity of their TF-IDF vectors. **Critical**: fit on train only, transform dev/test.
30. TF-IDF cosine similarity (character n-grams): `analyzer='char'`, `ngram_range=(3,5)`, `max_features=20000`
31. LSA cosine similarity: Apply TruncatedSVD (100 components) to the word TF-IDF vectors, then cosine similarity in LSA space
32. LSA cosine similarity (character n-grams): Same with char TF-IDF
33. WordNet Wu-Palmer similarity (max): For each hypothesis word, find the maximum Wu-Palmer similarity to any premise word via WordNet synsets. Average over hypothesis words.
34. WordNet Wu-Palmer similarity (avg): Average all pairwise Wu-Palmer similarities
35. WordNet Path similarity (max): Same methodology as Wu-Palmer but using path_similarity
36. WordNet Path similarity (avg)
37. WordNet Leacock-Chodorow similarity (max): lch_similarity averaged over best matches
38. WordNet Leacock-Chodorow similarity (avg)
39. Word Mover's Distance (WMD) approximation: Using pre-computed word vectors (if GloVe loaded) or using TF-IDF weighted centroid distance. Since closed mode but pre-trained embeddings are standard tools, GloVe 6B 100d can be used.
40. Sentence embedding cosine similarity: Average GloVe vectors per sentence, cosine similarity. Use GloVe 6B 100d (pre-trained, not task-specific data).
41. Sentence embedding cosine similarity (TF-IDF weighted): Weight each word's GloVe vector by its IDF weight before averaging.
42. Soft cosine similarity: Using word similarity matrix from GloVe embeddings and soft cosine measure
43. BM25 score (premise as document, hypothesis as query)
44. BM25 score (hypothesis as document, premise as query)
45. SIF embedding cosine similarity: Smooth Inverse Frequency (Arora et al. 2017) sentence embeddings using GloVe + principal component removal
46. SIF embedding L2 distance

**Tier 3: Negation and Contradiction Features (16 features)**

47. Negation presence in premise: 1 if any negation cue found (not, n't, no, never, neither, nobody, nothing, nowhere, hardly, scarcely, barely, seldom)
48. Negation presence in hypothesis: same check
49. Negation mismatch: XOR of premise/hypothesis negation presence
50. Negation count in premise
51. Negation count in hypothesis
52. Negation count difference: |neg_P - neg_H|
53. Antonym pair count: For each (premise_word, hypothesis_word) pair, check if they are antonyms in WordNet. Count total antonym pairs found.
54. Antonym pair presence: 1 if any antonym pair found, else 0
55. Number mismatch: 1 if numbers mentioned in premise and hypothesis differ (e.g., "16,000" vs "30,000")
56. Number overlap ratio: if both contain numbers, what fraction match
57. Named entity mismatch: 1 if named entities in hypothesis do not appear in premise (using spaCy NER)
58. Named entity overlap ratio: |NE_P ∩ NE_H| / max(|NE_H|, 1)
59. Sentiment polarity difference: |sentiment(P) - sentiment(H)| using TextBlob or VADER (both are general tools, not task-specific)
60. Sentiment subjectivity difference: |subjectivity(P) - subjectivity(H)|
61. Modal verb mismatch: presence of modal verbs (can, could, may, might, will, would, shall, should, must) -- XOR between P and H
62. Quantifier mismatch: presence of quantifiers (all, every, each, some, few, many, most, no, none, any) -- check if P and H have conflicting quantifiers

**Tier 4: Syntactic Features (20 features)**

63. POS tag distribution difference (premise vs hypothesis): For each of the 17 universal POS tags, compute |freq_P(tag) - freq_H(tag)|. This gives 17 features.
64-79. (17 POS difference features, as above: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X)
80. Dependency tree depth difference: |depth(P_tree) - depth(H_tree)| using spaCy dependency parse
81. SVO alignment score: Extract subject-verb-object triples from P and H (via dependency parsing), compute overlap of S, V, O components separately, average the three overlap scores
82. Root verb match: 1 if root verbs (ROOT dependency) of premise and hypothesis match (lemmatized), else 0

**Tier 5: Alignment-Based Features (12 features) -- Sultan et al. 2014 inspired**

Implement monolingual word alignment between premise and hypothesis words. The alignment algorithm:
- Step 1: Exact match alignment (same word in both, unaligned)
- Step 2: Lemma match alignment (same lemma)
- Step 3: WordNet synonym alignment
- Step 4: Contextual match alignment (same POS and neighboring aligned words)

From the alignment, extract:
83. Alignment coverage (premise): fraction of premise words that are aligned
84. Alignment coverage (hypothesis): fraction of hypothesis words that are aligned
85. Content word alignment coverage (premise): alignment coverage considering only content words
86. Content word alignment coverage (hypothesis)
87. Aligned pair count (total number of aligned word pairs)
88. Exact match aligned pairs / total aligned pairs
89. Lemma match aligned pairs / total aligned pairs
90. Synonym aligned pairs / total aligned pairs
91. Contextual match aligned pairs / total aligned pairs
92. Average alignment confidence: mean of alignment quality scores (exact=1.0, lemma=0.9, synonym=0.8, contextual=0.6)
93. Alignment symmetry: |coverage_P - coverage_H| (asymmetric alignment may indicate one-directional entailment)
94. Unaligned hypothesis content words count: number of hypothesis content words without alignment (high count suggests contradiction or novel information)

**Tier 6: Natural Logic Features (8 features) -- MacCartney & Manning 2007/2009 inspired**

For each aligned word pair from Tier 5, classify the lexical relation into one of MacCartney's 7 natural logic relations:
- = (equivalence): exact match or synonyms
- < (forward entailment): hypernym in premise, hyponym in hypothesis
- > (reverse entailment): hyponym in premise, hypernym in hypothesis
- ^ (alternation/negation): antonyms
- | (independence): unrelated
- ∨ (cover): co-hyponyms (share a common hypernym)

Then aggregate:
95. Fraction of aligned pairs with equivalence (=) relation
96. Fraction with forward entailment (<) relation
97. Fraction with reverse entailment (>) relation
98. Fraction with alternation/negation (^) relation
99. Fraction with independence (|) relation
100. Fraction with cover (∨) relation
101. "Entailment score": weighted combination favoring = and < relations
102. "Contradiction score": weighted combination favoring ^ relations

**Tier 7: Cross-Sentence Structural Features (10 features)**

103. Sentence count in premise
104. Sentence count in hypothesis
105. Sentence count ratio
106. Average word length in premise
107. Average word length in hypothesis
108. Word length ratio
109. Premise is a question: 1 if premise ends with ?
110. Hypothesis is a question: 1 if hypothesis ends with ?
111. Premise word count
112. Hypothesis word count

**Tier 8: Bag-of-Words Cross Features (dimensionality-reduced, ~100 features)**

113-212. TF-IDF of concatenated [premise; hypothesis] with TruncatedSVD reduction to 100 dimensions. This captures the joint topic/content space. Fit on training data only.

**Tier 9: Interaction Features (18 features)**

213. unigram_overlap * negation_mismatch (interaction: high overlap + negation flip = contradiction)
214. coverage_H * antonym_presence
215. jaccard_unigram * length_ratio
216. tfidf_cosine * negation_mismatch
217. alignment_coverage_H * contradiction_score
218. alignment_coverage_H * entailment_score
219. wordnet_wup_max * alignment_coverage_H
220. lsa_cosine * negation_count_diff
221. sif_cosine * antonym_presence
222. bleu4_forward * negation_mismatch
223. number_mismatch * tfidf_cosine
224. entity_mismatch * alignment_coverage_H
225. exact_match * length_ratio
226. substring_containment * hypothesis_word_count
227. modal_mismatch * tfidf_cosine
228. quantifier_mismatch * unigram_overlap
229. sentiment_diff * alignment_symmetry
230. root_verb_match * alignment_coverage_H

**Total feature vector dimension: ~280 features**

Note: Exact count depends on implementation. Core features: 28 + 18 + 16 + 20 + 12 + 8 + 10 + 100 + 18 = 230 base features + ~50 from POS distribution breakdown = ~280.

### 1.4 Classifier: Stacking Ensemble

**Base classifiers (Level 0):**

1. **XGBoost**:
   - No scaling needed (tree-based)
   - Hyperparameter search (RandomizedSearchCV, 5-fold stratified, 100 iterations):
     - n_estimators: [200, 500, 800, 1000, 1500]
     - max_depth: [3, 5, 7, 10, 12]
     - learning_rate: [0.01, 0.03, 0.05, 0.1, 0.15]
     - subsample: [0.6, 0.7, 0.8, 0.9, 1.0]
     - colsample_bytree: [0.6, 0.7, 0.8, 0.9, 1.0]
     - min_child_weight: [1, 3, 5, 7]
     - reg_alpha: [0, 0.01, 0.1, 1, 10]
     - reg_lambda: [1, 5, 10, 50]
     - gamma: [0, 0.1, 0.5, 1]
     - scale_pos_weight: [1.0] (near-balanced classes)
   - `eval_metric='logloss'`, `use_label_encoder=False`

2. **LightGBM**:
   - No scaling needed
   - Hyperparameter search:
     - n_estimators: [200, 500, 800, 1000, 1500]
     - max_depth: [-1, 5, 7, 10, 15] (-1 = no limit)
     - learning_rate: [0.01, 0.03, 0.05, 0.1]
     - num_leaves: [31, 63, 127, 255]
     - subsample: [0.6, 0.7, 0.8, 0.9, 1.0]
     - colsample_bytree: [0.6, 0.7, 0.8, 0.9, 1.0]
     - min_child_samples: [5, 10, 20, 50]
     - reg_alpha: [0, 0.01, 0.1, 1]
     - reg_lambda: [0, 0.01, 0.1, 1, 10]
   - `verbose=-1` to suppress output

3. **SVM-RBF**:
   - Preprocessing: StandardScaler (zero mean, unit variance) -- fitted on train only
   - Hyperparameter search (GridSearchCV, 5-fold stratified):
     - C: [0.01, 0.1, 1, 10, 100, 1000]
     - gamma: ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1]
   - `probability=True` for soft predictions

4. **Logistic Regression** (base learner):
   - Preprocessing: StandardScaler (same as SVM)
   - Hyperparameter search:
     - C: [0.001, 0.01, 0.1, 1, 10, 100]
     - solver: ['lbfgs', 'saga']
     - penalty: ['l2']
     - max_iter: 2000

**Meta-classifier (Level 1): Logistic Regression**
- Input: probability predictions from all 4 base classifiers (4 probability features if using predict_proba[:,1], or 8 if using both class probabilities)
- Use sklearn `StackingClassifier` with `cv=5` and `passthrough=False`
- Alternative experiment: `passthrough=True` to also pass original features to meta-learner
- Meta-learner hyperparameters: C=1.0, solver='lbfgs', max_iter=1000

**Training procedure:**
1. Extract all ~280 features for training data (this is the slowest step -- expect 30-60 minutes for 24K pairs with spaCy and WordNet)
2. The stacking classifier uses internal 5-fold CV to generate out-of-fold predictions from base classifiers, then trains the meta-learner on those predictions
3. Final evaluation on the held-out dev set
4. Save all fitted models using `joblib.dump`: ensemble model, TF-IDF vectorizers, SVD transformers, StandardScaler, GloVe vectors (or reference to file)

**Feature engineering speedup tips:**
- Use `spacy.pipe()` with `batch_size=256` and `n_process=2` for POS/dependency parsing
- Disable unnecessary spaCy pipeline components: only enable `tagger`, `parser`, `lemmatizer`, `ner`
- Pre-compute WordNet synsets for all unique words and cache in a dict
- Pre-compute GloVe vectors for all unique words and cache
- Consider parallelizing feature extraction with `joblib.Parallel` across feature groups

### 1.5 Creativity Justification (for Model Card and Presentation)

- **Monolingual word alignment for NLI**: Cite Sultan et al. (2014) "Back to Basics for Monolingual Alignment: Exploiting Word Similarity and Contextual Evidence", TAC. Their alignment-based NLI system achieved competitive performance on SICK and SNLI. Our alignment features (coverage, quality, symmetry) go well beyond simple overlap.
- **Natural logic features**: Cite MacCartney & Manning (2007) "Natural Logic for Textual Inference" and MacCartney & Manning (2009) "An extended model of natural logic", IWCS. Natural logic provides a compositional theory of entailment at the lexical level, and encoding these relations as features is a principled approach to NLI.
- **Stacking ensemble**: Cite Wolpert (1992) "Stacked Generalization" and the effectiveness of ensemble methods in NLI shared tasks.
- **Feature interaction terms**: Explicitly encoding negation-overlap interactions captures the key failure mode of overlap-based approaches (high overlap does not imply entailment when negation is present).
- **WordNet knowledge integration**: Using WordNet for synonym, antonym, hypernym, and co-hyponym detection goes beyond surface-level features. Cite Fellbaum (1998) "WordNet: An Electronic Lexical Database".
- **SIF sentence embeddings**: Cite Arora et al. (2017) "A Simple but Tough-to-Beat Baseline for Sentence Embeddings" -- provides robust sentence-level similarity without a neural model.

### 1.6 Soundness Checklist

- Feature scaling applied before SVM and Logistic Regression (StandardScaler fitted on train only)
- TF-IDF vectorizers and SVD transformers fitted on train only, transform applied to dev/test
- No data leakage: dev set never seen during training/feature fitting
- WordNet lookups handle missing synsets gracefully (many words have no synsets)
- Division-by-zero protection for all ratio features (use max(denominator, 1))
- Empty premise handling: placeholder text to avoid crashes
- Alignment algorithm handles edge cases (empty sentences, single-word sentences)
- Natural logic relation classification defaults to "independence" when no WordNet relation found
- Class balance is verified (~48/52 split, no special handling needed beyond default)
- All hyperparameter selection uses cross-validation on training data only
- GloVe embeddings are pre-trained general-purpose vectors, not task-specific data -- allowed in closed mode

---

## SOLUTION 2: CATEGORY B -- ESIM with WordNet Knowledge Enhancement (KIM-inspired)

### 2.1 Architecture Overview

Enhanced Sequential Inference Model (ESIM) with knowledge-enhanced inference. The model follows the standard ESIM pipeline: BiLSTM encoding of both premise and hypothesis, soft-attention cross-alignment, enhancement layer computing element-wise comparison of aligned vs original representations, composition BiLSTM, and pooling to fixed vectors. The key enhancement is a KIM-inspired (Knowledge-based Inference Model, Chen et al. 2018) injection of WordNet lexical relation features between aligned word pairs, concatenated into the comparison vectors before composition. Additionally, a character-level CNN handles out-of-vocabulary words for morphological robustness.

### 2.2 Data Preprocessing

1. **Word vocabulary**: Build from training data. Include all words appearing >= 2 times. Map rare words to `<UNK>`. Add `<PAD>` token. Expected vocabulary size: ~25,000-35,000 words.

2. **Character vocabulary**: All printable ASCII characters + `<PAD>` + `<UNK>`. Expected: ~98 characters.

3. **Word embeddings**: Load GloVe 840B 300d vectors. For each word in vocabulary:
   - If found in GloVe: use the pre-trained vector
   - If not found: initialize with uniform random in [-0.05, 0.05]
   - `<PAD>` token: zero vector
   - Embeddings are frozen for the first 5 epochs, then fine-tuned with lr=0.1x main lr

4. **Sequence length**:
   - Premise max length: 64 tokens (covers 99%+ of premises; avg=18.9 words)
   - Hypothesis max length: 32 tokens (covers 99%+ of hypotheses; avg=10.4 words)
   - Character max length per word: 16 characters
   - Pad shorter sequences with `<PAD>`, truncate longer ones

5. **Text normalization** (minimal):
   - Lowercase all text
   - Tokenize using spaCy or simple whitespace + punctuation splitting
   - Do NOT stem or lemmatize (GloVe vectors capture morphological variants)
   - Replace digits with `0` for better generalization (e.g., "2023" -> "0000")

6. **WordNet relation matrix (pre-computed)**:
   For each (premise_word_i, hypothesis_word_j) pair, pre-compute a 5-dimensional binary vector encoding WordNet relations:
   - [is_synonym, is_antonym, is_hypernym, is_hyponym, is_co_hyponym]
   - Use NLTK WordNet interface: `wn.synsets(word)`, check `lemma_names()`, `antonyms()`, `hypernyms()`, `hyponyms()`
   - Co-hyponym: two words share a common hypernym within 2 hops
   - Cache all lookups for efficiency
   - This produces a tensor of shape (batch, premise_len, hypothesis_len, 5)

7. **PyTorch Dataset**: Return `(premise_word_ids, premise_char_ids, hypothesis_word_ids, hypothesis_char_ids, wordnet_relation_matrix, label)` as tensors. Use `DataLoader` with `shuffle=True` for training, `batch_size=32`.

### 2.3 Model Architecture (Exact Specifications)

```
ESIM WITH KNOWLEDGE ENHANCEMENT

═══════════════════════════════════════════════════════
 INPUT ENCODING LAYER
═══════════════════════════════════════════════════════

Premise:  (batch, P_len=64)  word indices
Hypothesis: (batch, H_len=32)  word indices
Premise chars:  (batch, P_len, C_len=16)  char indices
Hypothesis chars: (batch, H_len, C_len=16)  char indices

1. Word Embedding:
   - nn.Embedding(vocab_size=35000, embedding_dim=300, padding_idx=0)
   - Initialize with GloVe 840B 300d
   - Output: premise_word_emb (batch, 64, 300)
             hypothesis_word_emb (batch, 32, 300)

2. Character-level CNN (shared for both P and H):
   - nn.Embedding(char_vocab_size=98, embedding_dim=8, padding_idx=0)
   - Reshape chars: (batch * seq_len, C_len=16, 8)
   - Conv1D: nn.Conv1d(in=8, out=50, kernel_size=5, padding=2) + ReLU
   - Max-pool over character dimension: (batch * seq_len, 50)
   - Reshape back: (batch, seq_len, 50)
   - Output: premise_char_emb (batch, 64, 50)
             hypothesis_char_emb (batch, 32, 50)

3. Concatenate word + char embeddings:
   - premise_emb = cat([premise_word_emb, premise_char_emb], dim=2)
     -> (batch, 64, 350)
   - hypothesis_emb = cat([hypothesis_word_emb, hypothesis_char_emb], dim=2)
     -> (batch, 32, 350)

4. Input Projection (optional, reduces dimension):
   - nn.Linear(350, 300) + ReLU + Dropout(0.2)
   - premise_proj (batch, 64, 300)
   - hypothesis_proj (batch, 32, 300)

5. Input Encoding BiLSTM:
   - nn.LSTM(input_size=300, hidden_size=300, num_layers=1,
             batch_first=True, bidirectional=True, dropout=0)
   - Output: premise_enc (batch, 64, 600) [300*2 bidirectional]
             hypothesis_enc (batch, 32, 600)

═══════════════════════════════════════════════════════
 LOCAL INFERENCE / CROSS-ATTENTION LAYER
═══════════════════════════════════════════════════════

6. Attention Weight Matrix:
   - e_ij = premise_enc_i^T * hypothesis_enc_j
   - attention_matrix = premise_enc @ hypothesis_enc.transpose(1,2)
     -> (batch, 64, 32)

7. Soft Alignment:
   - premise_attention = softmax(attention_matrix, dim=2) @ hypothesis_enc
     -> aligned_premise (batch, 64, 600)
     (each premise word attends to hypothesis)
   - hypothesis_attention = softmax(attention_matrix.transpose(1,2), dim=2) @ premise_enc
     -> aligned_hypothesis (batch, 32, 600)
     (each hypothesis word attends to premise)

═══════════════════════════════════════════════════════
 KNOWLEDGE-ENHANCED INFERENCE (KIM-INSPIRED)
═══════════════════════════════════════════════════════

8. WordNet Relation Integration:
   - Input: wordnet_relations (batch, P_len, H_len, 5) -- pre-computed
   - For each premise word i, compute weighted knowledge vector:
     k_P_i = sum_j(attention_weight_ij * wordnet_relations[i,j,:])
     -> (batch, 64, 5)
   - For each hypothesis word j, compute:
     k_H_j = sum_i(attention_weight_ji * wordnet_relations[i,j,:])
     -> (batch, 32, 5)
   - Project knowledge vectors:
     nn.Linear(5, 50) + ReLU
     -> k_P_proj (batch, 64, 50)
        k_H_proj (batch, 32, 50)

═══════════════════════════════════════════════════════
 ENHANCEMENT / COMPARISON LAYER
═══════════════════════════════════════════════════════

9. Enhancement (per original ESIM + knowledge):
   - For premise:
     m_P = cat([premise_enc,           # original (600)
                aligned_premise,        # aligned (600)
                premise_enc - aligned_premise,  # difference (600)
                premise_enc * aligned_premise,  # element-wise product (600)
                k_P_proj],              # knowledge (50)
               dim=2)
     -> (batch, 64, 2450)
   - For hypothesis:
     m_H = cat([hypothesis_enc,
                aligned_hypothesis,
                hypothesis_enc - aligned_hypothesis,
                hypothesis_enc * aligned_hypothesis,
                k_H_proj],
               dim=2)
     -> (batch, 32, 2450)

10. Enhancement Projection:
    - nn.Linear(2450, 300) + ReLU + Dropout(0.3)
    - m_P_proj (batch, 64, 300)
    - m_H_proj (batch, 32, 300)

═══════════════════════════════════════════════════════
 COMPOSITION LAYER
═══════════════════════════════════════════════════════

11. Composition BiLSTM:
    - nn.LSTM(input_size=300, hidden_size=300, num_layers=1,
              batch_first=True, bidirectional=True, dropout=0)
    - v_P = composition_output for premise (batch, 64, 600)
    - v_H = composition_output for hypothesis (batch, 32, 600)

═══════════════════════════════════════════════════════
 POOLING LAYER
═══════════════════════════════════════════════════════

12. Pooling (applied separately to premise and hypothesis):
    - v_P_avg = average_pool(v_P, mask) -> (batch, 600)
    - v_P_max = max_pool(v_P, mask) -> (batch, 600)
    - v_H_avg = average_pool(v_H, mask) -> (batch, 600)
    - v_H_max = max_pool(v_H, mask) -> (batch, 600)
    - v = cat([v_P_avg, v_P_max, v_H_avg, v_H_max], dim=1)
      -> (batch, 2400)

═══════════════════════════════════════════════════════
 CLASSIFICATION MLP
═══════════════════════════════════════════════════════

13. MLP Classifier:
    - nn.Linear(2400, 512) + Tanh + Dropout(0.3)
    - nn.Linear(512, 256) + Tanh + Dropout(0.3)
    - nn.Linear(256, 1) + Sigmoid
    - Output: P(entailment)

Total trainable parameters estimate:
  - Word embeddings: 35000 * 300 = 10.5M (frozen initially)
  - Char embeddings: 98 * 8 = 784
  - Char CNN: 8*50*5 + 50 = 2,050
  - Input projection: 350*300 + 300 = 105,300
  - Input BiLSTM: 4 * (300*300 + 300*300 + 300) * 2 = ~1.44M
  - Knowledge projection: 5*50 + 50 = 300
  - Enhancement projection: 2450*300 + 300 = 735,300
  - Composition BiLSTM: 4 * (300*300 + 300*300 + 300) * 2 = ~1.44M
  - MLP: 2400*512 + 512 + 512*256 + 256 + 256*1 + 1 = ~1.36M
  - Total trainable (excl. frozen embeddings): ~5.1M
  - Total with embeddings: ~15.6M
```

### 2.4 Training Procedure

**Optimizer**: Adam (not AdamW -- following original ESIM)
- lr: 4e-4 (initial)
- betas: (0.9, 0.999)
- eps: 1e-8

**Learning rate schedule**: ReduceLROnPlateau
- mode: 'max' (monitoring macro_f1)
- factor: 0.5
- patience: 3
- min_lr: 1e-6

**Word embedding fine-tuning schedule**:
- Epochs 1-5: word embeddings frozen (only char CNN, BiLSTMs, MLP trained)
- Epochs 6+: word embeddings unfrozen with lr = 0.1x main lr (using separate param group)

**Loss function**: Binary Cross-Entropy Loss
- `nn.BCELoss()` or equivalently `nn.BCEWithLogitsLoss()` (if removing final sigmoid and using logits)
- No class weighting needed (near-balanced dataset)

**Batch size**: 32

**Epochs**: Max 40, with early stopping (patience=7 on dev macro_f1)

**Regularization**:
- Dropout as specified: 0.2 after input projection, 0.3 after enhancement projection, 0.3 in MLP layers
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)`
- Early stopping on dev macro_f1

**Data augmentation** (applied during training only):
- Premise-hypothesis swap augmentation: For entailment pairs (label=1), with probability 0.1, swap premise and hypothesis. This is valid since many entailment pairs have bidirectional entailment (paraphrases). NOTE: Only apply this if validation shows improvement; not all entailments are symmetric.
- Word dropout: With probability 0.1, replace random words with `<UNK>` during training (improves generalization and OOV robustness)

**Checkpointing**: Save model `state_dict` at the epoch with best dev macro_f1

**Hardware**: GPU required (V100/A100/T4 on Colab or university cluster). Estimated training time: 2-4 hours for 40 epochs.

### 2.5 Hyperparameter Sensitivity Analysis

Run ablation over:
- Hidden size: [150, 200, 300] (for both BiLSTMs)
- Char CNN filters: [25, 50, 100]
- Char CNN kernel size: [3, 5, 7]
- Dropout: [0.2, 0.3, 0.4, 0.5]
- Learning rate: [1e-4, 2e-4, 4e-4, 8e-4]
- Batch size: [16, 32, 64]
- With/without character CNN
- With/without WordNet knowledge features
- With/without word dropout augmentation
- Premise max length: [48, 64, 80]
- Hypothesis max length: [24, 32, 48]

### 2.6 Creativity Justification

- **ESIM (Enhanced Sequential Inference Model)**: Cite Chen et al. (2017) "Enhanced LSTM for Natural Language Inference", ACL. ESIM is the foundational neural architecture for NLI that introduced cross-attention alignment + enhancement comparison. Our implementation faithfully follows this architecture.
- **KIM (Knowledge-based Inference Model)**: Cite Chen et al. (2018) "Neural Natural Language Inference Models Enhanced with External Knowledge", ACL. KIM demonstrated that injecting WordNet lexical relations into the attention-weighted comparison vectors significantly improves NLI performance. Our knowledge enhancement layer directly follows the KIM methodology.
- **Character-level CNN for morphological robustness**: Cite Kim et al. (2016) "Character-Aware Neural Language Models" and Santos & Zadrozny (2014). The char CNN handles OOV words and captures morphological patterns (e.g., "un-" prefix for negation, "-ly" suffix for adverbs) that are important for NLI.
- **GloVe 840B embeddings**: Cite Pennington et al. (2014) "GloVe: Global Vectors for Word Representation". The 840B variant provides the broadest vocabulary coverage.
- **Combination novelty**: The combination of ESIM + KIM knowledge features + character CNN is not a standard off-the-shelf configuration. The KIM paper used a different base architecture; adapting their knowledge injection mechanism into ESIM is a creative integration of two complementary papers.

### 2.7 Soundness Checklist

- Cross-attention is computed between premise and hypothesis encodings (not self-attention, which would make this a transformer and violate Category B)
- No transformer components used anywhere: all attention is bilinear dot-product attention in the cross-attention layer, which is standard LSTM-based attention
- BiLSTM uses bidirectional processing but no multi-head self-attention
- GloVe embeddings are frozen initially to prevent catastrophic forgetting of pre-trained knowledge
- WordNet relation matrix is pre-computed and stored as tensors (not computed during forward pass) for efficiency
- Masking is applied correctly in pooling layers to ignore `<PAD>` tokens
- Gradient clipping prevents exploding gradients in BiLSTMs
- Character CNN padding preserves sequence length
- Binary classification with single sigmoid output is appropriate for 2-class NLI
- Enhancement layer follows exact ESIM formulation [a; ã; a-ã; a*ã] plus knowledge features

---

## SOLUTION 3: CATEGORY C (BACKUP) -- DeBERTa-v3-base with Hypothesis-Only Adversarial Debiasing

### 3.1 Architecture Overview

Fine-tune DeBERTa-v3-base as a cross-encoder for NLI: concatenate premise and hypothesis as `[CLS] premise [SEP] hypothesis [SEP]`, encode with DeBERTa, and classify the [CLS] representation. The key creative element is a hypothesis-only adversarial debiasing mechanism: a gradient reversal layer (GRL) on an auxiliary head that tries to predict the label from hypothesis-only representations. This forces the main classifier to rely on premise-hypothesis interaction rather than hypothesis-only shortcuts (e.g., negation words in hypothesis correlating with contradiction). This addresses the well-known annotation artifacts problem in NLI (Gururangan et al. 2018, McCoy et al. 2019).

### 3.2 Data Preprocessing

1. **Tokenization**: Use DeBERTa-v3-base tokenizer from HuggingFace (`microsoft/deberta-v3-base`).
   - Input format: `[CLS] premise [SEP] hypothesis [SEP]`
   - `token_type_ids` to distinguish premise tokens (0) from hypothesis tokens (1)

2. **Max sequence length**: 128 tokens. Rationale: premise avg ~19 words (~25 subword tokens), hypothesis avg ~10 words (~14 subword tokens), total ~39 words (~55 subword tokens including special tokens). 128 provides ample headroom for longer examples while being memory-efficient.

3. **For the adversarial debiasing head**: Also tokenize hypothesis-only input: `[CLS] hypothesis [SEP]`. Max length: 48 tokens.

4. **HuggingFace tokenizer call**:
   ```python
   # Main input
   tokenizer(premise, hypothesis, max_length=128, padding='max_length',
             truncation=True, return_tensors='pt')
   # Hypothesis-only input (for adversarial head)
   tokenizer(hypothesis, max_length=48, padding='max_length',
             truncation=True, return_tensors='pt')
   ```

5. **PyTorch Dataset**: Return `(main_input_ids, main_attention_mask, main_token_type_ids, hypo_input_ids, hypo_attention_mask, label)`.

### 3.3 Model Architecture (Exact Specifications)

```
CROSS-ENCODER WITH ADVERSARIAL DEBIASING

═══════════════════════════════════════════════════════
 MAIN ENCODER (Premise + Hypothesis)
═══════════════════════════════════════════════════════

Input: [CLS] premise [SEP] hypothesis [SEP]
       input_ids: (batch, 128)
       attention_mask: (batch, 128)
       token_type_ids: (batch, 128)

1. DeBERTa-v3-base Encoder:
   - Load from microsoft/deberta-v3-base (HuggingFace)
   - 12 layers, 768 hidden dim, 12 attention heads
   - ~86M parameters
   - Extract [CLS] token representation from last hidden state
   - cls_main = last_hidden_state[:, 0, :]  -> (batch, 768)

2. Classification Head:
   - nn.Dropout(0.1)
   - nn.Linear(768, 256) + GELU + nn.Dropout(0.1)
   - nn.Linear(256, 1) + Sigmoid
   - Output: P(entailment) -> (batch, 1)

═══════════════════════════════════════════════════════
 ADVERSARIAL HYPOTHESIS-ONLY HEAD
═══════════════════════════════════════════════════════

3. Hypothesis-Only Encoder (shared DeBERTa weights):
   - Input: [CLS] hypothesis [SEP]
     hypo_input_ids: (batch, 48)
     hypo_attention_mask: (batch, 48)
   - Pass through SAME DeBERTa encoder
   - cls_hypo = last_hidden_state[:, 0, :]  -> (batch, 768)

4. Gradient Reversal Layer:
   - GradientReversalLayer(lambda_grl)
   - During forward: identity function
   - During backward: multiply gradients by -lambda_grl
   - Applied to cls_hypo

5. Hypothesis-Only Prediction Head:
   - nn.Linear(768, 128) + ReLU + nn.Dropout(0.2)
   - nn.Linear(128, 1) + Sigmoid
   - Output: P(entailment from hypothesis only)
   - Loss: BCE on this prediction
   - The GRL reverses gradients, so the encoder is trained to make
     hypothesis-only representations LESS predictive of the label

═══════════════════════════════════════════════════════
 COMBINED LOSS
═══════════════════════════════════════════════════════

6. Total Loss:
   L = L_main + lambda_adv * L_hypo_only
   where:
     L_main = BCE(main_prediction, label)
     L_hypo_only = BCE(hypo_only_prediction, label)
     lambda_adv = adversarial loss weight (hyperparameter)

   Note: The GRL already reverses the gradient direction for L_hypo_only
   with respect to the encoder. The lambda_adv controls the STRENGTH
   of the adversarial signal. The lambda_grl in GRL controls the
   GRADIENT SCALING.

   Final effective behavior: encoder minimizes L_main while being pushed
   by the reversed L_hypo_only gradient to produce representations that
   are NOT predictive from hypothesis alone.
```

### 3.4 Training Procedure

**Phase 1: Standard fine-tuning (epochs 1-3)**
- Freeze adversarial head (lambda_grl = 0, lambda_adv = 0)
- Train only: DeBERTa encoder + main classification head
- Optimizer: AdamW
  - lr: 2e-5
  - weight_decay: 0.01
  - betas: (0.9, 0.999)
  - eps: 1e-6
- Loss: BCE only on main head
- Purpose: stabilize fine-tuning before introducing adversarial signal

**Phase 2: Adversarial debiasing (epochs 4-15+)**
- Activate adversarial head
- lambda_grl: ramp linearly from 0.0 to target over 2 epochs
  - target lambda_grl: 0.1 (hyperparameter to tune in [0.01, 0.05, 0.1, 0.2, 0.5])
- lambda_adv: 1.0 (the GRL handles the gradient scaling)
- Continue training with composite loss
- Optimizer: AdamW with discriminative learning rates:
  - DeBERTa layers 0-5: lr=1e-5
  - DeBERTa layers 6-11: lr=2e-5
  - Classification head: lr=5e-5
  - Adversarial head: lr=1e-4

**Learning rate schedule**: Linear warmup (first 10% of total steps) then linear decay to 0
- Warmup steps: max(100, int(0.1 * total_steps))

**Batch size**: 16

**Epochs**: Max 15, early stopping patience=3 on dev macro_f1

**Gradient accumulation**: 4 steps (effective batch size = 64)

**Mixed precision**: fp16 training via `torch.cuda.amp` (GradScaler + autocast)

**Regularization**:
- Dropout: 0.1 in classification head, 0.2 in adversarial head
- Weight decay: 0.01 (applied to all parameters except bias and LayerNorm)
- Gradient clipping: max_norm=1.0
- Early stopping on dev macro_f1

**Checkpointing**: Save best model (by dev macro_f1) and last model

**Hardware**: GPU required. Estimated training time: 1-2 hours on V100/A100 for 15 epochs.

### 3.5 Hyperparameter Sensitivity Analysis

Run ablation over:
- Base model: [deberta-v3-base, deberta-v3-large (if GPU memory allows), roberta-base (as comparison)]
- Max sequence length: [96, 128, 160]
- Learning rate: [1e-5, 2e-5, 3e-5, 5e-5]
- Batch size (effective): [32, 64, 128]
- lambda_grl: [0, 0.01, 0.05, 0.1, 0.2, 0.5] (0 = no adversarial debiasing)
- Weight decay: [0.01, 0.05, 0.1]
- Epochs: [5, 10, 15, 20]
- Warmup ratio: [0.05, 0.1, 0.2]
- With/without adversarial debiasing
- With/without discriminative learning rates

### 3.6 Creativity Justification

- **Hypothesis-only adversarial debiasing**: Cite McCoy et al. (2019) "Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference", ACL, and Gururangan et al. (2018) "Annotation Artifacts in Natural Language Inference Data", NAACL. These papers showed that NLI models learn hypothesis-only shortcuts. Our GRL-based debiasing directly addresses this by adversarially training the encoder to be less reliant on hypothesis-only cues.
- **Gradient reversal layer for debiasing**: Cite Ganin & Lempitsky (2015) "Unsupervised Domain Adaptation by Backpropagation", ICML, for the GRL mechanism. Adapting GRL from domain adaptation to NLI debiasing is a creative transfer.
- **DeBERTa-v3 as base model**: Cite He et al. (2021) "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing". DeBERTa-v3 outperforms BERT-base and RoBERTa-base on NLI benchmarks due to its disentangled attention mechanism and enhanced mask decoder.
- **Discriminative fine-tuning**: Cite Howard & Ruder (2018) "Universal Language Model Fine-tuning for Text Classification" (ULMFiT).
- **Combined contribution**: Standard DeBERTa fine-tuning for NLI is vanilla Category C. Our adversarial debiasing mechanism is a principled addition that (a) improves robustness, (b) demonstrates understanding of NLI-specific challenges, and (c) cites relevant literature.

### 3.7 Soundness Checklist

- DeBERTa-v3-base is a pre-trained language model, not external task-specific data -- allowed in closed mode
- Cross-encoder (concatenated input) is the standard and most effective architecture for NLI with transformers
- Hypothesis-only encoding uses the SAME DeBERTa weights, creating the adversarial tension (if separate weights were used, the adversarial signal would not affect the main encoder)
- GRL lambda is ramped gradually to avoid destabilizing early training
- lambda_grl = 0 reduces to standard fine-tuning (safe fallback if adversarial hurts)
- Discriminative LR prevents catastrophic forgetting of pre-trained knowledge in lower layers
- Max sequence length 128 is sufficient for this dataset (avg combined ~39 words)
- fp16 training is safe with GradScaler to prevent underflow
- Weight decay is not applied to bias and LayerNorm parameters (standard practice)

---

## EVALUATION PLAN (3 marks for Evaluation criterion)

This section targets full marks on the "Evaluation" criterion by going significantly beyond Codabench benchmarking. All evaluations are NLI-specific.

### 4.1 Quantitative Evaluation (for each of the two submitted solutions)

1. **Full metric suite on dev set**: Report all 8 metrics from the scorer (accuracy, macro precision, macro recall, macro_f1, weighted precision, weighted recall, weighted F1, MCC). Present in a comparison table.

2. **Confusion matrices**: For each solution, plot a 2x2 confusion matrix heatmap (using seaborn/matplotlib). Report TP, FP, TN, FN counts and rates. Labels: 0 = not entailed, 1 = entailed.

3. **Per-class precision, recall, F1**: Break down performance on class 0 (not entailed) vs class 1 (entailed). Identify whether the model has a bias toward predicting entailment or non-entailment.

4. **Calibration analysis**: Plot reliability diagrams (predicted probability vs observed frequency) for each model. Use `sklearn.calibration.calibration_curve`. This shows whether the model's probability outputs are well-calibrated (important for NLI where confidence matters).

5. **ROC curves and AUC**: Plot ROC curves for each solution. Report AUC. For Category A, this requires probability outputs from the ensemble.

6. **Precision-Recall curves**: More informative than ROC for understanding the trade-off. Report area under PR curve.

### 4.2 Statistical Significance Testing

7. **McNemar's test**: Compare each solution against its category baseline:
   - Cat A vs SVM baseline (macro_f1=0.5846)
   - Cat B vs LSTM baseline (macro_f1=0.6603)
   - Cat C vs BERT baseline (macro_f1=0.8198)
   Baseline predictions available at `/Users/kumar/Documents/University/Year3/NLU/project/baseline_extracted/nlu_bundle-feature-unified-local-scorer/baseline/25_DEV_NLI.csv`.
   McNemar's test operates on the 2x2 contingency table of (correct/incorrect) between two classifiers. Report the chi-squared statistic and p-value. A p-value < 0.05 means the improvement is statistically significant, which the rubric explicitly requires for the competitive performance marks.

8. **Bootstrap confidence intervals**: Compute 95% confidence intervals for macro_f1 and MCC via bootstrap resampling (1000 iterations) on the dev set. This quantifies uncertainty in the performance estimates. Report: point estimate, 95% CI lower, 95% CI upper.

9. **Paired bootstrap test between solutions**: If submitting A+B, compare the two solutions against each other using paired bootstrap to determine which is statistically superior and on what subsets.

### 4.3 Error Analysis (NLI-Specific)

10. **Error analysis by inference type**: Manually categorize 100 random dev errors into:
    - Lexical overlap heuristic failures (high overlap but wrong label)
    - Negation/contradiction misses
    - Numerical reasoning failures
    - World knowledge required
    - Coreference resolution required
    - Quantifier reasoning failures
    - Temporal reasoning failures
    - Syntactic variation (active/passive, nominalization)
    Present as a bar chart with counts per error type.

11. **Hypothesis-only baseline test (NLI-specific)**:
    Train a simple logistic regression classifier using ONLY hypothesis features (TF-IDF of hypothesis for Cat A, or hypothesis-only encoding for Cat B/C). Report its macro_f1. If it exceeds 0.55, there are annotation artifacts. Compare the main model's performance on examples where the hypothesis-only baseline is correct vs incorrect -- a robust model should perform well on BOTH subsets, not just the "easy" examples where hypothesis-only cues work.

12. **Lexical overlap analysis (NLI-specific)**:
    Bin dev examples by Jaccard word overlap between premise and hypothesis:
    - Low overlap (Jaccard < 0.1): report macro_f1
    - Medium overlap (0.1 <= Jaccard < 0.3): report macro_f1
    - High overlap (Jaccard >= 0.3): report macro_f1
    A robust NLI model should not be fooled by high overlap (e.g., "A man bites a dog" has high overlap with "A dog bites a man" but different meaning).

13. **Negation handling analysis (NLI-specific)**:
    Separate dev examples into:
    - Pairs with negation in hypothesis but not premise
    - Pairs with negation in premise but not hypothesis
    - Pairs with negation in both
    - Pairs with no negation
    Report macro_f1 for each group. Models that handle negation well should have consistent performance.

14. **Length-based analysis**:
    Group dev pairs by:
    - Premise length (short <10 words, medium 10-25, long >25)
    - Hypothesis length (short <5 words, medium 5-12, long >12)
    - Length ratio bins
    Report macro_f1 per group. Identify if model struggles with very short or very long inputs.

### 4.4 Ablation Studies

15. **Feature ablation for Category A**: Train the ensemble with feature group subsets removed:
    - All features (baseline)
    - Without Tier 1 (lexical overlap)
    - Without Tier 2 (semantic similarity)
    - Without Tier 3 (negation/contradiction)
    - Without Tier 4 (syntactic)
    - Without Tier 5 (alignment)
    - Without Tier 6 (natural logic)
    - Without Tier 8 (BoW cross features)
    - Without Tier 9 (interaction features)
    - Only Tier 1 + 2 (surface features only)
    - Only Tier 5 + 6 (alignment + natural logic only)
    Report macro_f1 for each ablation. Present as a waterfall chart showing feature importance.

16. **Architecture ablation for Category B**:
    - Full model (ESIM + char CNN + WordNet knowledge)
    - Without character CNN (word embeddings only)
    - Without WordNet knowledge enhancement (standard ESIM)
    - Without enhancement layer (no [a-ã, a*ã] comparison)
    - With avg pooling only (no max pooling)
    - With max pooling only (no avg pooling)
    - Single-layer MLP (instead of 3-layer)
    - Unidirectional LSTM (instead of BiLSTM)
    Report macro_f1 for each.

17. **Adversarial debiasing ablation for Category C**:
    - Full model (DeBERTa + adversarial debiasing)
    - Without adversarial head (standard DeBERTa fine-tuning)
    - Different lambda_grl values: 0, 0.01, 0.05, 0.1, 0.2, 0.5
    - Hypothesis-only performance of debiased vs non-debiased model (should decrease with debiasing)
    Report macro_f1 and hypothesis-only accuracy for each.

### 4.5 Cross-Solution Analysis

18. **Agreement analysis**: Compute Cohen's kappa between solutions. Analyze the dev pairs where the two solutions disagree. Are there systematic patterns? Does one model handle negation better while the other handles paraphrases better?

19. **Ensemble of solutions**: What happens if we majority-vote or average probabilities across A, B, and C? (Not for submission, but demonstrates analytical thinking and could be mentioned in the poster.)

20. **Error overlap**: Venn diagram of errors made by each solution. What fraction of errors are shared? Unique to each?

---

## MODEL CARD PLAN (13 marks: 3 formatting + 6 informativeness + 4 accuracy)

Each model card must use the template at `/Users/kumar/Documents/University/Year3/NLU/project/archive_extracted/COMP34812_modelcard_template.md` and be generated via the Jinja-based notebook.

### 5.1 Template Fields (For Each Model Card)

The template has these fields: `model_id`, `model_summary`, `model_description`, `developers`, `base_model_repo`, `base_model_paper`, `model_type`, `model_architecture`, `language`, `base_model`, `training_data`, `hyperparameters`, `speeds_sizes_times`, `testing_data`, `testing_metrics`, `results`, `hardware_requirements`, `software`, `bias_risks_limitations`, `additional_information`.

### 5.2 Model Card for Category A (Feature-Rich Stacking Ensemble)

- **model_id**: `username1-username2-NLI-CatA`
- **model_summary**: "A traditional machine learning system for natural language inference that uses ~280 hand-crafted features spanning lexical overlap, semantic similarity, negation detection, syntactic structure, monolingual word alignment (Sultan et al. 2014), and natural logic relations (MacCartney & Manning 2009), classified by a stacking ensemble of XGBoost, LightGBM, SVM, and Logistic Regression with a Logistic Regression meta-learner."
- **model_description**: 2-3 paragraphs covering: the NLI task definition (given premise and hypothesis, determine entailment), the feature engineering approach (9 tiers with counts), the alignment-based features (cite Sultan 2014), natural logic features (cite MacCartney 2007/2009), the stacking ensemble architecture, and the key design rationale (combining surface overlap with deeper semantic and knowledge-based features).
- **developers**: Full names
- **base_model_repo**: "N/A (no pre-trained model used, though GloVe 6B 100d word vectors are used for semantic similarity features)"
- **base_model_paper**: List all key papers: Sultan et al. (2014), MacCartney & Manning (2007, 2009), Bowman et al. (2015), Arora et al. (2017), Pennington et al. (2014), Wolpert (1992)
- **model_type**: "Supervised classification (stacking ensemble)"
- **model_architecture**: "Stacking ensemble (XGBoost + LightGBM + SVM-RBF + Logistic Regression, meta-learner: Logistic Regression) on ~280 hand-crafted NLI features including lexical overlap, semantic similarity, negation/contradiction, syntactic, word alignment, and natural logic features."
- **language**: "English"
- **base_model**: "N/A"
- **training_data**: "24,432 premise-hypothesis pairs from the COMP34812 NLI training set. Premises average 18.9 words (max 281), hypotheses average 10.4 words (max 45). Labels: 48.2% non-entailment (0), 51.8% entailment (1). Near-balanced binary classification."
- **hyperparameters**: List EXACT final hyperparameters for each base classifier and meta-learner after tuning. Include feature extraction parameters (TF-IDF max_features, SVD components, GloVe dimensions).
- **speeds_sizes_times**: "Feature extraction: ~45 minutes for 24K pairs on CPU. XGBoost training: ~5 minutes. Full ensemble training (with 5-fold stacking): ~30 minutes. Total pipeline: ~75 minutes. Model size on disk: ~50MB (includes fitted TF-IDF, SVD, scaler, ensemble)."
- **testing_data**: "6,735 premise-hypothesis pairs from the COMP34812 NLI development set. Performance was also assessed via 5-fold stratified cross-validation on the training set."
- **testing_metrics**: "Macro F1-score (primary), Matthews Correlation Coefficient (MCC), accuracy, macro precision, macro recall, weighted F1. Statistical significance assessed via McNemar's test against SVM baseline (macro_f1=0.5846)."
- **results**: Complete table of all 8 metrics on dev set. Include comparison to SVM baseline. Include McNemar's p-value. Include key ablation results (feature importance ranking).
- **hardware_requirements**: "CPU only. RAM: 16GB recommended (WordNet and GloVe loading). No GPU required."
- **software**: "Python 3.9+, scikit-learn==1.3.x, xgboost==2.0.x, lightgbm==4.1.x, spacy==3.7.x (en_core_web_sm), nltk==3.8.x (WordNet corpus), numpy, pandas, gensim (for GloVe loading, optional)"
- **bias_risks_limitations**: "(1) Feature engineering assumes English text -- WordNet, function word list, spaCy models are all English-specific. (2) Natural logic features rely on WordNet coverage, which is incomplete for informal/slang text. (3) Alignment algorithm may produce poor alignments for very short premises or hypotheses. (4) TF-IDF and SVD are fitted on training data only; domain shift in test data may reduce effectiveness. (5) The model cannot capture complex multi-hop reasoning patterns that require compositional understanding. (6) GloVe vectors have known biases (gender, racial) that propagate into similarity features."
- **additional_information**: Feature importance ranking from XGBoost/LightGBM. Ablation study findings showing which feature tiers are most impactful. Hypothesis-only baseline comparison.

### 5.3 Model Card for Category B (ESIM + KIM Knowledge Enhancement)

- **model_id**: `username1-username2-NLI-CatB`
- **model_summary**: "A deep learning system for natural language inference based on the Enhanced Sequential Inference Model (ESIM, Chen et al. 2017) with Knowledge-enhanced Inference (KIM, Chen et al. 2018). The model uses BiLSTM encoding, cross-attention alignment, WordNet lexical relation injection, composition BiLSTM, and pooling for classification. Character-level CNN embeddings handle out-of-vocabulary words."
- **model_description**: 2-3 paragraphs covering: the NLI task, the ESIM architecture (encoding -> cross-attention -> enhancement -> composition -> pooling -> classification), the KIM knowledge enhancement (WordNet relations injected into comparison vectors), the character CNN for OOV robustness, and the rationale for combining these approaches.
- **model_architecture**: "ESIM: Word Embedding (GloVe 840B 300d) + Char CNN (50d) -> BiLSTM (300 hidden, bidirectional, 600d output) -> Cross-Attention Alignment -> Knowledge Enhancement (WordNet 5d relations projected to 50d) -> Enhancement Comparison [a; ã; a-ã; a*ã; knowledge] -> Projection (300d) -> Composition BiLSTM (300 hidden, 600d) -> Max+Avg Pooling (2400d) -> MLP (512 -> 256 -> 1) with Sigmoid."
- **base_model**: "GloVe 840B 300d pre-trained word vectors (Pennington et al. 2014)"
- **base_model_repo**: "https://nlp.stanford.edu/projects/glove/"
- **base_model_paper**: "Chen et al. (2017) Enhanced LSTM for Natural Language Inference; Chen et al. (2018) Neural NLI Models Enhanced with External Knowledge; Pennington et al. (2014) GloVe"
- **hyperparameters**: EXACT values: word_emb_dim=300, char_emb_dim=8, char_cnn_filters=50, char_cnn_kernel=5, bilstm_hidden=300, bilstm_layers=1, knowledge_dim=5, knowledge_proj_dim=50, mlp_hidden_1=512, mlp_hidden_2=256, dropout_input=0.2, dropout_enhancement=0.3, dropout_mlp=0.3, optimizer=Adam, lr=4e-4, batch_size=32, max_epochs=40, early_stopping_patience=7, gradient_clip=10.0, premise_max_len=64, hypothesis_max_len=32, char_max_len=16
- **speeds_sizes_times**: "Training: ~3 hours on V100 GPU for 40 epochs (early stopping typically at epoch 20-30). Model size: ~60MB (state_dict). Inference: ~500 pairs/second on GPU."
- **training_data**: Same as Cat A description.
- **testing_data**: Same as Cat A description, with comparison to LSTM baseline (macro_f1=0.6603).
- **testing_metrics**: Same metrics as Cat A, with McNemar's test against LSTM baseline.
- **results**: Complete metric table. Comparison to LSTM baseline. Ablation results (with/without char CNN, with/without knowledge enhancement).
- **hardware_requirements**: "GPU required (NVIDIA GPU with >= 8GB VRAM). Tested on V100 (16GB). CPU inference possible but slow (~50 pairs/second)."
- **software**: "Python 3.9+, PyTorch==2.1.x, nltk==3.8.x (WordNet), numpy, spacy==3.7.x (for tokenization)"
- **bias_risks_limitations**: "(1) GloVe embeddings encode societal biases from training corpora that may affect NLI predictions (e.g., gender stereotypes in similarity computations). (2) WordNet coverage is limited for informal text, slang, and domain-specific terminology. (3) Character CNN helps with OOV but cannot fully compensate for words absent from GloVe's vocabulary. (4) Fixed sequence length truncation may lose information from very long premises (max 281 words vs 64 token limit). (5) Cross-attention is shallow (single-layer) and may miss complex multi-step reasoning. (6) Binary NLI formulation loses the distinction between contradiction and neutral."
- **additional_information**: Attention weight visualization examples showing which hypothesis words attend to which premise words. Knowledge feature impact analysis.

### 5.4 Model Card for Category C (DeBERTa with Adversarial Debiasing, if submitted)

- **model_id**: `username1-username2-NLI-CatC`
- **model_summary**: "A transformer-based NLI system using DeBERTa-v3-base fine-tuned as a cross-encoder with a hypothesis-only adversarial debiasing mechanism (gradient reversal layer) to prevent reliance on annotation artifacts and hypothesis-only shortcuts."
- **base_model**: "microsoft/deberta-v3-base"
- **base_model_repo**: "https://huggingface.co/microsoft/deberta-v3-base"
- **base_model_paper**: "He et al. (2021) DeBERTaV3; McCoy et al. (2019) Right for the Wrong Reasons; Ganin & Lempitsky (2015) Domain Adaptation by Backpropagation"
- **model_architecture**: "DeBERTa-v3-base (12 layers, 768 hidden, 12 heads) as cross-encoder with [CLS] premise [SEP] hypothesis [SEP] input. Classification head: Dropout(0.1) -> Linear(768, 256) + GELU + Dropout(0.1) -> Linear(256, 1) + Sigmoid. Adversarial head: GRL(lambda) -> Linear(768, 128) + ReLU + Dropout(0.2) -> Linear(128, 1) + Sigmoid, operating on hypothesis-only [CLS] representation."
- **hyperparameters**: EXACT values: max_seq_len=128, hypo_max_len=48, lr=2e-5, weight_decay=0.01, batch_size=16, grad_accum=4, effective_batch=64, max_epochs=15, early_stopping=3, warmup_ratio=0.1, lambda_grl=0.1, fp16=True, gradient_clip=1.0
- **speeds_sizes_times**: "Training: ~1.5 hours on V100 GPU for 15 epochs. Model size: ~350MB. Inference: ~300 pairs/second on GPU."
- **hardware_requirements**: "GPU required (NVIDIA GPU with >= 8GB VRAM)."
- All other fields with equivalent specificity.

### 5.5 Accuracy of Model Cards (4 marks)

The highest-weighted single criterion. Every claim in the model card must be verifiable from the code:
- Hyperparameters in the card must match the code EXACTLY
- Architecture descriptions must match the model class definitions
- Training procedure must match the training loop
- Results must match the scorer output
- Do a final review pass: read each sentence of each model card and cross-check it against the code

---

## CODE ORGANIZATION PLAN (3 marks Organisation + 3 marks Completeness)

### 6.1 Directory Structure

```
submission/
├── README.md                          # Project overview, how to run, attributions, AI tool declaration
├── requirements.txt                   # All Python dependencies with versions
├── Group_n_A.csv                      # Category A predictions on test set
├── Group_n_B.csv                      # Category B predictions on test set (or Group_n_C.csv)
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb    # Data exploration, statistics, visualizations
│   ├── 02_CatA_Training.ipynb            # Category A: feature engineering + ensemble training
│   ├── 03_CatB_Training.ipynb            # Category B: ESIM+KIM training
│   ├── 04_CatC_Training.ipynb            # Category C: DeBERTa training (if submitted)
│   ├── 05_Evaluation.ipynb               # All evaluation: confusion matrices, McNemar's, ablation, error analysis
│   ├── 06_Demo_CatA.ipynb                # Demo: load model, run inference on input CSV, produce predictions
│   ├── 07_Demo_CatB.ipynb                # Demo: load model, run inference
│   └── 08_Model_Card_Generation.ipynb    # Generate model cards from template
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                     # Data loading, preprocessing, tokenization
│   ├── feature_engineering.py            # All NLI feature extraction for Cat A (~280 features)
│   ├── alignment.py                      # Sultan et al. monolingual word alignment implementation
│   ├── natural_logic.py                  # MacCartney natural logic relation classification
│   ├── wordnet_utils.py                  # WordNet similarity, antonym, hypernym lookups with caching
│   ├── models/
│   │   ├── __init__.py
│   │   ├── esim_kim.py                   # Cat B: ESIM + KIM PyTorch model definition
│   │   ├── deberta_adversarial.py        # Cat C: DeBERTa + adversarial debiasing model
│   │   └── gradient_reversal.py          # GRL layer implementation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_ensemble.py             # Cat A training logic
│   │   ├── train_esim.py                 # Cat B training loop
│   │   └── train_deberta.py              # Cat C training loop
│   └── evaluation/
│       ├── __init__.py
│       └── eval_utils.py                 # Confusion matrix, McNemar's, bootstrap CI, error analysis
│
├── models/                               # Saved trained models (or link to cloud if >10MB)
│   ├── cat_a_ensemble.joblib             # Category A: fitted sklearn stacking pipeline
│   ├── cat_a_tfidf_word.joblib           # Category A: fitted word TF-IDF vectorizer
│   ├── cat_a_tfidf_char.joblib           # Category A: fitted char TF-IDF vectorizer
│   ├── cat_a_svd_word.joblib             # Category A: fitted SVD (word)
│   ├── cat_a_svd_char.joblib             # Category A: fitted SVD (char)
│   ├── cat_a_svd_cross.joblib            # Category A: fitted SVD (cross BoW)
│   ├── cat_a_scaler.joblib               # Category A: fitted StandardScaler
│   ├── cat_b_esim_kim.pt                 # Category B: PyTorch state_dict
│   ├── cat_b_vocab.json                  # Category B: word and char vocabulary mappings
│   └── cat_c_deberta/                    # Category C: HuggingFace save format
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer/
│
├── model_cards/
│   ├── model_card_catA.md                # Generated model card for Category A
│   └── model_card_catB.md                # Generated model card for Category B (or C)
│
├── data/                                  # Reference to data location (not included in submission)
│   └── glove.840B.300d.txt               # Or cloud link in README
│
└── poster/
    └── poster.pdf                        # A1 landscape or 16:9 PowerPoint PDF
```

### 6.2 README.md Contents

1. **Project overview**: "COMP34812 NLU Coursework -- Natural Language Inference (NLI) Track. Given a premise and a hypothesis, predict whether the hypothesis is entailed by the premise (binary classification)."
2. **Team members**: Full names
3. **Solutions summary**: One paragraph per solution describing approach and key results
4. **How to install**: `pip install -r requirements.txt` with exact Python version. `python -m spacy download en_core_web_sm`. `nltk.download('wordnet')`.
5. **How to train**: Point to each training notebook with clear instructions
6. **How to run demo/inference**: "Given a CSV file with columns `premise` and `hypothesis`, the demo code loads the trained model and produces a predictions CSV with a single column of 0/1 labels."
7. **How to evaluate**: Point to evaluation notebook
8. **Code structure**: Directory tree with descriptions
9. **Model storage**: If models >10MB (likely for Cat B and C), provide OneDrive/cloud link
10. **Data attribution**: "Training and evaluation data provided by the COMP34812 teaching team. GloVe word vectors from Stanford NLP (Pennington et al. 2014). WordNet from Princeton University (Fellbaum 1998)."
11. **Code attribution**: Cite any code adapted from other sources with URLs
12. **Use of Generative AI Tools**: Declaration as required by Section VI of coursework spec

### 6.3 Documentation Standards

- Every notebook has markdown cells explaining what each code block does
- Every function has a docstring with parameters, return type, and brief description
- Every hyperparameter is defined as a named constant at the top of the notebook/file (not magic numbers)
- Inline comments for non-obvious logic (e.g., alignment algorithm steps, natural logic classification rules)
- Type hints in function signatures
- Feature names are stored in a list and logged alongside the feature matrix for traceability

### 6.4 Completeness Checklist

- All trained models saved and loadable
- Demo notebooks: given a CSV path, produce predictions end-to-end
- No hardcoded file paths (use relative paths or configurable paths via constants at top)
- `requirements.txt` includes ALL dependencies with pinned versions
- All random seeds set for reproducibility: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, `sklearn` uses `random_state=42` everywhere
- Demo code includes `!pip install` cells for any non-standard packages
- WordNet and spaCy model downloads handled gracefully (with error messages if missing)

---

## DECISION FRAMEWORK: SELECTING THE BEST TWO SOLUTIONS

### 7.1 Decision Criteria (in priority order)

1. **Dev set macro_f1 gap over baseline**: The primary metric. Compute the gap between each solution and its category baseline:
   - Cat A gap = solution_A_f1 - 0.5846 (SVM baseline)
   - Cat B gap = solution_B_f1 - 0.6603 (LSTM baseline)
   - Cat C gap = solution_C_f1 - 0.8198 (BERT baseline)

2. **Statistical significance**: Run McNemar's test. If a solution's improvement over its baseline is NOT statistically significant (p > 0.05), it risks getting 0/3 on competitive performance for that solution.

3. **MCC**: As a secondary metric. Higher MCC improvement is impressive.

4. **Creativity headroom**: A solution with higher architectural creativity scores better on the 3-mark creativity criterion, even if its absolute F1 is slightly lower.

### 7.2 Expected Decision Matrix

| Metric | Cat A Expected | Cat B Expected | Cat C Expected |
|--------|---------------|----------------|----------------|
| Dev macro_f1 | 0.65-0.75 | 0.78-0.85 | 0.83-0.88 |
| Baseline macro_f1 | 0.5846 | 0.6603 | 0.8198 |
| Expected gap | +0.07-0.17 | +0.12-0.19 | +0.01-0.06 |
| McNemar significant? | Very likely | Very likely | Possible |
| Creativity score | Very high | Very high | High |

### 7.3 Decision Rules

- **If Cat A > 0.64 AND Cat B > 0.72**: Submit A + B. Both have meaningful gaps over baselines, both are highly creative, and this combination is the most distinctive. A + B is the **default recommendation**.
- **If Cat C > Cat B by more than 0.05 F1 AND Cat C beats BERT baseline significantly (p < 0.05)**: Consider A + C instead, since competitive performance marks may be higher.
- **If Cat B fails to beat 0.68 (close to LSTM baseline)**: Discard B, submit A + C.
- **If Cat A fails to beat 0.60**: This would be surprising given the feature richness, but discard A and submit B + C.
- **Default recommendation**: A + B (highest combined creativity, largest expected gaps over baselines, most distinctive combination, best demonstration of NLI understanding).

---

## PREDICTION GENERATION WORKFLOW (Test Data March 24)

### 8.1 Pre-Test-Data Preparation (Before March 24)

1. **All three models fully trained and validated** on dev set
2. **Prediction pipelines tested**: For each model, write and test a `predict(csv_path) -> predictions` function using the dev CSV as a dry run. Verify that the output matches the expected format.
3. **Output format verified**: Single column, integers (0 or 1). For dev, expect 6,735 rows (matching the solution file). For test, the number of rows will match the test CSV. No header needed.
4. **Scorer tested**: Run the local scorer and verify the scores match expectations:
   ```bash
   # Using the local scorer
   python -c "
   from local_scorer.metrics import compute_metrics
   solution = [int(line.strip()) for line in open('NLU_SharedTask_NLI_dev.solution')]
   prediction = [int(line.strip()) for line in open('my_predictions.csv')]
   results = compute_metrics(solution, prediction)
   for name, value in results:
       print(f'{name}: {value:.4f}')
   "
   ```

### 8.2 Test Day Workflow (March 24)

1. **Download test CSV** from Canvas. Expected format: `premise,hypothesis` (no label column).
2. **Verify format**: Check column names, row count, no label column.
3. **Run Category A prediction**:
   - Load saved TF-IDF vectorizers, SVD transformers, scaler, ensemble model
   - Extract all ~280 features from test pairs (same pipeline as training)
   - Generate predictions
   - Save as `Group_n_A.csv` (single column of 0/1 integers)
4. **Run Category B prediction**:
   - Load saved ESIM-KIM model and vocabulary
   - Tokenize and encode test pairs (word + char)
   - Pre-compute WordNet relation matrices for test pairs
   - Run forward pass with `model.eval()` and `torch.no_grad()`
   - Apply threshold (0.5) to sigmoid output
   - Save as `Group_n_B.csv`
5. **Run Category C prediction** (if applicable):
   - Load saved DeBERTa model
   - Tokenize test pairs with DeBERTa tokenizer
   - Run forward pass
   - Save as `Group_n_C.csv`
6. **Verify predictions**: Check row count matches test set, values are 0 or 1 only, no NaN/null, no header
7. **Select best two** based on the decision framework (using dev performance as proxy)
8. **Final sanity check**: Ensure prediction file names follow convention `Group_n_X.csv`

### 8.3 Submission Assembly (Before March 27 18:00)

1. Package into a single zip file:
   - Prediction CSVs (2 files)
   - All notebooks (training + evaluation + demo)
   - Source code modules
   - Trained models (or cloud links if >10MB in README)
   - Model cards (2 markdown files)
   - Poster PDF
   - README.md
   - requirements.txt
2. Verify zip file size (<10MB excluding models, or use cloud links)
3. Upload to Canvas before deadline

---

## RISK MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Cat A feature engineering takes too long (spaCy + WordNet on 24K pairs) | Medium | Medium | Start with Tier 1-2 features only (fast, no spaCy needed). Add tiers incrementally. Pre-compute and cache spaCy annotations. Use `spacy.pipe()` with batching. |
| Cat A alignment features are too slow | Medium | Low | Implement simple alignment (exact + lemma match only, skip contextual step). Cache WordNet synsets. |
| Cat B ESIM fails to converge | Low | High | Start without knowledge enhancement (standard ESIM). Add char CNN and knowledge features after baseline works. Simplify to 1-layer MLP. |
| Cat B WordNet relation matrix computation is slow | Medium | Low | Pre-compute and save to disk. Cache synset lookups in dict. Limit to first 3 synsets per word. |
| Cat C DeBERTa does not beat BERT baseline significantly | Medium | High | This is why we build all 3: if Cat C only marginally beats BERT, submit A+B instead. Also try roberta-large as alternative base model. |
| Adversarial debiasing (Cat C) destabilizes training | Medium | Medium | Start with lambda_grl=0 (standard fine-tuning). Ramp gradually. If still unstable, disable and submit standard DeBERTa. |
| Hypothesis-only bias in dataset | Medium | Medium | The hypothesis-only baseline test (Eval #11) detects this. If bias is severe, Cat C's adversarial debiasing addresses it. For Cat A, negation/contradiction features partially handle this. |
| GPU training too slow | Low | Medium | Cat B: 2-4 hours expected. Cat C: 1-2 hours expected. Use Colab Pro or university cluster. Reduce batch size or sequence length if needed. |
| Model files exceed 10MB | Likely for B,C | Low | Upload to OneDrive, include link in README as instructed in coursework spec. Cat A models (~50MB) also need cloud storage. |
| GloVe 840B download fails or too large | Low | Medium | Fallback: use GloVe 6B 300d (smaller, still effective). Or use 6B 100d for Cat A features. |
| Empty/very short premise in test data | Low | Low | Already handled: placeholder text, division-by-zero guards. Tested with training data (min premise = 0 words). |
| Cat A overfitting on interaction features | Medium | Medium | Use cross-validation to select features. Remove interaction features if they do not improve CV score. Regularize ensemble (higher reg_alpha/reg_lambda). |

---

## TIMELINE

| Day | Tasks |
|-----|-------|
| Day 1 (Mar 22) | Set up project structure. Implement data loading and preprocessing. Begin Cat A feature engineering (Tiers 1-3: lexical overlap, semantic similarity, negation). |
| Day 2 (Mar 23) | Complete Cat A features (Tiers 4-9: syntactic, alignment, natural logic, cross, interaction). Train initial Cat A ensemble. Begin Cat B data pipeline (tokenization, vocab building, WordNet relation matrix, DataLoader). |
| Day 3 (Mar 24) | **Test data released.** Finish Cat B ESIM+KIM model implementation and begin training. Begin Cat C DeBERTa implementation. Generate Cat A test predictions immediately. |
| Day 4 (Mar 25) | Complete Cat B training and tuning. Complete Cat C training. Generate all test predictions. Run full evaluation suite (confusion matrices, McNemar's, ablation). |
| Day 5 (Mar 26) | Error analysis (hypothesis-only baseline, lexical overlap analysis, negation analysis). Select best two solutions. Write model cards. Polish code and documentation. |
| Day 6 (Mar 27, before 18:00) | Final review of model card accuracy. Package submission zip. Upload to Canvas. |

---

## PAPERS TO CITE (Complete Reference List)

### Core NLI Papers
1. Bowman, S., Angeli, G., Potts, C., & Manning, C. (2015). "A large annotated corpus for learning natural language inference." EMNLP.
2. Williams, A., Nangia, N., & Bowman, S. (2018). "A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference." NAACL (MultiNLI).
3. Dagan, I., Glickman, O., & Magnini, B. (2005). "The PASCAL Recognising Textual Entailment Challenge." MLCW (original RTE formulation).

### Category A Citations
4. Sultan, M. A., Bethard, S., & Sumner, T. (2014). "Back to Basics for Monolingual Alignment: Exploiting Word Similarity and Contextual Evidence." TAC.
5. Sultan, M. A., Bethard, S., & Sumner, T. (2015). "DLS@CU: Sentence Similarity from Word Alignment and Semantic Vector Composition." SemEval.
6. MacCartney, B. & Manning, C. D. (2007). "Natural Logic for Textual Inference." ACL Workshop on Textual Entailment and Paraphrasing.
7. MacCartney, B. & Manning, C. D. (2009). "An Extended Model of Natural Logic." IWCS.
8. MacCartney, B. (2009). "Natural Language Inference." PhD Thesis, Stanford University.
9. Arora, S., Liang, Y., & Ma, T. (2017). "A Simple but Tough-to-Beat Baseline for Sentence Embeddings." ICLR.
10. Fellbaum, C. (1998). "WordNet: An Electronic Lexical Database." MIT Press.
11. Wolpert, D. (1992). "Stacked Generalization." Neural Networks.
12. Pennington, J., Socher, R., & Manning, C. D. (2014). "GloVe: Global Vectors for Word Representation." EMNLP.

### Category B Citations
13. Chen, Q., Zhu, X., Ling, Z., Wei, S., Jiang, H., & Inkpen, D. (2017). "Enhanced LSTM for Natural Language Inference." ACL.
14. Chen, Q., Zhu, X., Ling, Z., Inkpen, D., & Wei, S. (2018). "Neural Natural Language Inference Models Enhanced with External Knowledge." ACL.
15. Kim, S., Kang, I., & Kwak, N. (2019). "Semantic Sentence Matching with Densely-Connected Recurrent and Co-Attentive Information." AAAI.
16. Santos, C. D. & Zadrozny, B. (2014). "Learning Character-level Representations for Part-of-Speech Tagging." ICML.
17. Kim, Y., Jernite, Y., Sontag, D., & Rush, A. M. (2016). "Character-Aware Neural Language Models." AAAI.
18. Conneau, A., Kiela, D., Schwenk, H., Barrault, L., & Bordes, A. (2017). "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data." EMNLP (InferSent -- uses BiLSTM+max pooling, related architecture).

### Category C Citations
19. He, P., Gao, J., & Chen, W. (2021). "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing." arXiv:2111.09543.
20. Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL.
21. Liu, Y., Ott, M., Goyal, N., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." arXiv:1907.11692.
22. Howard, J. & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." ACL (ULMFiT -- discriminative LR).

### Adversarial Debiasing & Annotation Artifacts
23. McCoy, R. T., Pavlick, E., & Linzen, T. (2019). "Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference." ACL.
24. Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S., & Smith, N. A. (2018). "Annotation Artifacts in Natural Language Inference Data." NAACL.
25. Ganin, Y. & Lempitsky, V. (2015). "Unsupervised Domain Adaptation by Backpropagation." ICML (Gradient Reversal Layer).
26. Belinkov, Y., Poliak, A., Shieber, S., Van Durme, B., & Rush, A. (2019). "On Adversarial Removal of Hypothesis-Only Bias in Natural Language Inference." Joint Conference on Lexical and Computational Semantics.
27. Clark, C., Yatskar, M., & Zettlemoyer, L. (2019). "Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases." EMNLP.

### Evaluation & Analysis
28. Khosla, P., et al. (2020). "Supervised Contrastive Learning." NeurIPS.
29. Naik, A., Ravichander, A., Sadeh, N., Rose, C., & Neubig, G. (2018). "Stress Test Evaluation for Natural Language Inference." COLING.

---

## APPENDIX A: GRADIENT REVERSAL LAYER IMPLEMENTATION

```python
import torch
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(torch.nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
```

---

## APPENDIX B: MONOLINGUAL WORD ALIGNMENT ALGORITHM (Sultan et al. 2014 simplified)

```python
def align_words(premise_tokens, hypothesis_tokens, premise_lemmas, hypothesis_lemmas):
    """
    Align words between premise and hypothesis using cascaded matching.
    Returns list of (premise_idx, hypothesis_idx, alignment_type) tuples.

    Alignment types: 'exact', 'lemma', 'synonym', 'contextual'
    """
    alignments = []
    aligned_p = set()
    aligned_h = set()

    # Step 1: Exact match (case-insensitive)
    for i, p_word in enumerate(premise_tokens):
        for j, h_word in enumerate(hypothesis_tokens):
            if i not in aligned_p and j not in aligned_h:
                if p_word.lower() == h_word.lower():
                    alignments.append((i, j, 'exact'))
                    aligned_p.add(i)
                    aligned_h.add(j)

    # Step 2: Lemma match
    for i, p_lemma in enumerate(premise_lemmas):
        for j, h_lemma in enumerate(hypothesis_lemmas):
            if i not in aligned_p and j not in aligned_h:
                if p_lemma.lower() == h_lemma.lower():
                    alignments.append((i, j, 'lemma'))
                    aligned_p.add(i)
                    aligned_h.add(j)

    # Step 3: WordNet synonym match
    for i, p_word in enumerate(premise_tokens):
        if i in aligned_p:
            continue
        p_synsets = wn.synsets(p_word.lower())
        p_synonyms = set()
        for syn in p_synsets:
            for lemma in syn.lemmas():
                p_synonyms.add(lemma.name().lower())
        for j, h_word in enumerate(hypothesis_tokens):
            if j not in aligned_h:
                if h_word.lower() in p_synonyms:
                    alignments.append((i, j, 'synonym'))
                    aligned_p.add(i)
                    aligned_h.add(j)
                    break

    # Step 4: Contextual match (neighboring aligned words + same POS)
    # For each unaligned pair, check if their neighbors are aligned
    for i, p_word in enumerate(premise_tokens):
        if i in aligned_p:
            continue
        for j, h_word in enumerate(hypothesis_tokens):
            if j in aligned_h:
                continue
            # Check if neighbors are aligned
            neighbor_aligned = False
            for di in [-1, 1]:
                for dj in [-1, 1]:
                    ni, nj = i + di, j + dj
                    if (ni, nj) in [(a[0], a[1]) for a in alignments]:
                        neighbor_aligned = True
            if neighbor_aligned:
                # Additional POS check would go here
                alignments.append((i, j, 'contextual'))
                aligned_p.add(i)
                aligned_h.add(j)
                break

    return alignments
```

---

## APPENDIX C: KEY DATA FILES AND PATHS

- `/Users/kumar/Documents/University/Year3/NLU/project/training_extracted/training_data/NLI/train.csv` - Primary training data: 24,432 premise-hypothesis pairs with labels (columns: premise, hypothesis, label)
- `/Users/kumar/Documents/University/Year3/NLU/project/training_extracted/training_data/NLI/dev.csv` - Development evaluation data: 6,736 rows (columns: premise, hypothesis, label)
- `/Users/kumar/Documents/University/Year3/NLU/project/baseline_extracted/nlu_bundle-feature-unified-local-scorer/local_scorer/reference_data/NLU_SharedTask_NLI_dev.solution` - Gold standard solution: 6,735 rows of 0/1 labels (1 fewer than dev.csv; verify alignment)
- `/Users/kumar/Documents/University/Year3/NLU/project/baseline_extracted/nlu_bundle-feature-unified-local-scorer/baseline/25_DEV_NLI.csv` - Baseline predictions: columns are [index, reference, SVM, LSTM, BERT]. Use SVM/LSTM/BERT columns for McNemar's tests.
- `/Users/kumar/Documents/University/Year3/NLU/project/baseline_extracted/nlu_bundle-feature-unified-local-scorer/local_scorer/metrics.py` - Scorer implementation: defines all metric functions (macro_f1, MCC, etc.)
- `/Users/kumar/Documents/University/Year3/NLU/project/archive_extracted/COMP34812_modelcard_template.md` - Model card Jinja template for generating model cards

---

## APPENDIX D: DATASET STATISTICS SUMMARY

| Statistic | Train | Dev |
|-----------|-------|-----|
| Total pairs | 24,432 | 6,736 |
| Label 0 (not entailed) | 11,784 (48.2%) | 3,258 (48.4%) |
| Label 1 (entailed) | 12,648 (51.8%) | 3,478 (51.6%) |
| Premise avg word count | 18.9 | ~19 |
| Premise max word count | 281 | ~280 |
| Premise min word count | 0 | ~0 |
| Hypothesis avg word count | 10.4 | ~10 |
| Hypothesis max word count | 45 | ~45 |
| Hypothesis min word count | 1 | ~1 |
| Evaluation rows (solution file) | N/A | 6,735 |

Note: The dev.csv has 6,736 data rows but the solution file has 6,735 rows. This 1-row discrepancy must be investigated before generating predictions. Possible causes: (a) the solution file excludes one row, or (b) off-by-one in line counting. The baseline CSV (25_DEV_NLI.csv) has 6,737 lines (1 header + 6,736 data rows), suggesting all dev rows are evaluated. Verify by checking if the last row of dev.csv is included in the solution.
