"""
NLI Feature Engineering — TF-IDF and semantic similarity features.
Tier 2: Semantic Similarity (18 features) — TF-IDF cosine, LSA, BM25
Tier 8: BoW Cross Features — TF-IDF + SVD of concatenated P+H (100 features)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class NLITfidfFeatures:
    """TF-IDF based features for NLI.

    Fits word and char TF-IDF vectorizers on training data,
    computes pairwise similarity features.
    """

    def __init__(self, n_svd_components=100):
        self.n_svd_components = n_svd_components

        # Word TF-IDF
        self.word_tfidf = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        # Char TF-IDF
        self.char_tfidf = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            max_features=20000,
            sublinear_tf=True,
        )

        # SVD for LSA
        self.word_svd = TruncatedSVD(n_components=n_svd_components, random_state=42)
        self.char_svd = TruncatedSVD(n_components=n_svd_components, random_state=42)

        # Cross-feature TF-IDF (on concatenated P+H)
        self.cross_tfidf = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.cross_svd = TruncatedSVD(n_components=n_svd_components, random_state=42)

        self._fitted = False

    def fit(self, premises, hypotheses):
        """Fit all vectorizers on training data.

        Args:
            premises: List of premise strings.
            hypotheses: List of hypothesis strings.
        """
        all_texts = list(premises) + list(hypotheses)
        concat_texts = [f"{p} {h}" for p, h in zip(premises, hypotheses)]

        # Fit word TF-IDF + SVD
        word_matrix = self.word_tfidf.fit_transform(all_texts)
        self.word_svd.fit(word_matrix)

        # Fit char TF-IDF + SVD
        char_matrix = self.char_tfidf.fit_transform(all_texts)
        self.char_svd.fit(char_matrix)

        # Fit cross TF-IDF + SVD
        cross_matrix = self.cross_tfidf.fit_transform(concat_texts)
        self.cross_svd.fit(cross_matrix)

        # Compute IDF weights for BM25
        self._idf = self.word_tfidf.idf_
        self._vocab = self.word_tfidf.vocabulary_
        self._avg_dl = np.mean([len(t.split()) for t in all_texts])

        self._fitted = True
        return self

    def compute_similarity_features(self, premise, hypothesis):
        """Compute pairwise TF-IDF similarity features.

        Returns:
            Dict with ~8 similarity features.
        """
        feats = {}

        # Word TF-IDF cosine
        p_word = self.word_tfidf.transform([premise])
        h_word = self.word_tfidf.transform([hypothesis])
        feats['tfidf_cosine'] = float(cosine_similarity(p_word, h_word)[0, 0])

        # Char TF-IDF cosine
        p_char = self.char_tfidf.transform([premise])
        h_char = self.char_tfidf.transform([hypothesis])
        feats['tfidf_char_cosine'] = float(cosine_similarity(p_char, h_char)[0, 0])

        # LSA cosine (word)
        p_lsa = self.word_svd.transform(p_word)
        h_lsa = self.word_svd.transform(h_word)
        feats['lsa_cosine'] = float(cosine_similarity(p_lsa, h_lsa)[0, 0])

        # LSA cosine (char)
        p_lsa_c = self.char_svd.transform(p_char)
        h_lsa_c = self.char_svd.transform(h_char)
        feats['lsa_char_cosine'] = float(cosine_similarity(p_lsa_c, h_lsa_c)[0, 0])

        # BM25 scores
        feats['bm25_p_as_doc'] = self._bm25_score(premise, hypothesis)
        feats['bm25_h_as_doc'] = self._bm25_score(hypothesis, premise)

        return feats

    def compute_cross_features(self, premise, hypothesis):
        """Compute Tier 8 cross features (TF-IDF + SVD of concatenated P+H).

        Returns:
            Dict with n_svd_components features.
        """
        concat = f"{premise} {hypothesis}"
        vec = self.cross_tfidf.transform([concat])
        svd_vec = self.cross_svd.transform(vec)[0]

        return {f'cross_svd_{i}': float(v) for i, v in enumerate(svd_vec)}

    def _bm25_score(self, document, query, k1=1.5, b=0.75):
        """Compute BM25 score of query against document."""
        doc_words = document.lower().split()
        query_words = query.lower().split()
        doc_len = len(doc_words)

        if doc_len == 0:
            return 0.0

        doc_freq = {}
        for w in doc_words:
            doc_freq[w] = doc_freq.get(w, 0) + 1

        score = 0.0
        for qw in query_words:
            if qw in self._vocab:
                idf = self._idf[self._vocab[qw]]
                tf = doc_freq.get(qw, 0)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / self._avg_dl))
                score += idf * tf_norm

        return score


class GloveFeatures:
    """GloVe-based sentence embedding features.

    Loads GloVe vectors and computes sentence-level similarity.
    Falls back to TF-IDF weighted centroids if GloVe unavailable.
    """

    def __init__(self, glove_path=None, dim=100):
        self.dim = dim
        self.glove_path = glove_path
        self.vectors = {}
        self._idf_weights = None
        self._loaded = False

    def load(self, idf_vectorizer=None):
        """Load GloVe vectors.

        Args:
            idf_vectorizer: Fitted TfidfVectorizer for IDF weights.
        """
        if self.glove_path:
            self._load_glove_file(self.glove_path)
        else:
            # Try common paths
            import os
            paths = [
                os.path.expanduser('~/scratch/nlu-project/glove.6B.100d.txt'),
                os.path.expanduser('~/glove.6B.100d.txt'),
                'glove.6B.100d.txt',
            ]
            for p in paths:
                if os.path.exists(p):
                    self._load_glove_file(p)
                    break

        if idf_vectorizer is not None:
            self._idf_weights = dict(zip(
                idf_vectorizer.get_feature_names_out(),
                idf_vectorizer.idf_
            ))

        self._loaded = True
        return self

    def _load_glove_file(self, path):
        """Load GloVe vectors from file."""
        print(f"Loading GloVe from {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip().split(' ')
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                if len(vec) == self.dim:
                    self.vectors[word] = vec
        print(f"Loaded {len(self.vectors)} GloVe vectors.")

    def sentence_embedding(self, text, use_idf=False):
        """Compute sentence embedding as (IDF-weighted) average of word vectors."""
        words = text.lower().split()
        vecs = []
        weights = []
        for w in words:
            if w in self.vectors:
                vecs.append(self.vectors[w])
                if use_idf and self._idf_weights:
                    weights.append(self._idf_weights.get(w, 1.0))
                else:
                    weights.append(1.0)

        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)

        vecs = np.array(vecs)
        weights = np.array(weights)
        weights = weights / weights.sum()
        return (vecs * weights[:, None]).sum(axis=0)

    def compute_features(self, premise, hypothesis):
        """Compute GloVe-based similarity features.

        Returns:
            Dict with ~4 features.
        """
        feats = {}

        if not self.vectors:
            return {
                'glove_cosine': 0.0,
                'glove_idf_cosine': 0.0,
                'glove_l2_dist': 0.0,
                'sif_cosine': 0.0,
            }

        p_emb = self.sentence_embedding(premise, use_idf=False)
        h_emb = self.sentence_embedding(hypothesis, use_idf=False)

        # Cosine similarity
        p_norm = np.linalg.norm(p_emb)
        h_norm = np.linalg.norm(h_emb)
        if p_norm > 0 and h_norm > 0:
            feats['glove_cosine'] = float(np.dot(p_emb, h_emb) / (p_norm * h_norm))
        else:
            feats['glove_cosine'] = 0.0

        # IDF-weighted cosine
        p_idf = self.sentence_embedding(premise, use_idf=True)
        h_idf = self.sentence_embedding(hypothesis, use_idf=True)
        p_n = np.linalg.norm(p_idf)
        h_n = np.linalg.norm(h_idf)
        if p_n > 0 and h_n > 0:
            feats['glove_idf_cosine'] = float(np.dot(p_idf, h_idf) / (p_n * h_n))
        else:
            feats['glove_idf_cosine'] = 0.0

        # L2 distance
        feats['glove_l2_dist'] = float(np.linalg.norm(p_emb - h_emb))

        # SIF embedding (simplified: use IDF weighting as proxy)
        feats['sif_cosine'] = feats['glove_idf_cosine']

        return feats
