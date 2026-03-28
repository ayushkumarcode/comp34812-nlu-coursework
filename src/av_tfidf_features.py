"""
Char n-gram TF-IDF + SVD features (group 3) and cosine similarity.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class CharNgramTFIDF:
    """Char n-gram TF-IDF followed by SVD to get dense features."""

    def __init__(self, ngram_range=(3, 5), max_features=10000, n_components=100):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit(self, texts):
        """Fit on all training texts (both columns combined)."""
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf_matrix)
        self._fitted = True
        return self

    def transform(self, texts):
        """Transform texts to dense SVD features -> (n_texts, n_components)."""
        tfidf_matrix = self.vectorizer.transform(texts)
        return self.svd.transform(tfidf_matrix)

    def transform_to_dict(self, text):
        """Transform one text, return as dict with tfidf_svd_0, tfidf_svd_1, etc."""
        vec = self.transform([text])[0]
        return {f'tfidf_svd_{i}': float(v) for i, v in enumerate(vec)}


class CosineSimFeatures:
    """Pairwise cosine similarity features using raw TF-IDF vectors.

    Computes cosine similarity at char 3-gram, 4-gram, and 5-gram levels.
    """

    def __init__(self, max_features=10000):
        self.max_features = max_features
        self.vectorizers = {}
        for n in [3, 4, 5]:
            self.vectorizers[n] = TfidfVectorizer(
                analyzer='char',
                ngram_range=(n, n),
                max_features=max_features,
                sublinear_tf=True,
            )
        self._fitted = False

    def fit(self, texts):
        """Fit vectorizers on all training texts.

        Args:
            texts: List of all training text strings.
        """
        for n, vec in self.vectorizers.items():
            vec.fit(texts)
        self._fitted = True
        return self

    def compute_similarities(self, text_1, text_2):
        """Compute cosine similarities for a text pair.

        Args:
            text_1: First text string.
            text_2: Second text string.

        Returns:
            Dict with 'cosine_char3', 'cosine_char4', 'cosine_char5'.
        """
        feats = {}
        for n, vec in self.vectorizers.items():
            v1 = vec.transform([text_1])
            v2 = vec.transform([text_2])
            sim = cosine_similarity(v1, v2)[0, 0]
            feats[f'cosine_char{n}'] = float(sim)
        return feats
