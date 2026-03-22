"""
NLI Category A — Full feature extraction pipeline.
Combines all feature tiers into a single feature matrix.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.nli_feature_engineering import (
    extract_basic_features,
    interaction_features,
)
from src.nli_tfidf_features import NLITfidfFeatures, GloveFeatures


class NLIFeatureExtractor:
    """Complete NLI Category A feature extraction pipeline.

    Handles:
    - Basic features (Tiers 1, 3, 7)
    - spaCy features (Tiers 4, 5, 6) — optional
    - TF-IDF similarity features (Tier 2)
    - Cross TF-IDF+SVD features (Tier 8)
    - GloVe features (Tier 2 supplement) — optional
    - Interaction features (Tier 9)
    """

    def __init__(self, use_spacy=True, use_glove=False, n_svd_components=100):
        self.use_spacy = use_spacy
        self.use_glove = use_glove
        self.tfidf = NLITfidfFeatures(n_svd_components=n_svd_components)
        self.glove = GloveFeatures() if use_glove else None
        self.nlp = None
        self._fitted = False
        self._feature_names = None

    def fit(self, df):
        """Fit TF-IDF vectorizers on training data.

        Args:
            df: Training DataFrame with premise, hypothesis columns.
        """
        premises = list(df['premise'])
        hypotheses = list(df['hypothesis'])
        print(f"Fitting TF-IDF on {len(premises)} pairs...")
        self.tfidf.fit(premises, hypotheses)

        if self.use_glove and self.glove:
            self.glove.load(idf_vectorizer=self.tfidf.word_tfidf)

        self._fitted = True
        return self

    def transform(self, df, show_progress=True):
        """Extract all features for a DataFrame of premise-hypothesis pairs.

        Args:
            df: DataFrame with premise, hypothesis columns.
            show_progress: If True, show progress bar.

        Returns:
            numpy array of shape (n_pairs, n_features), list of feature names.
        """
        n = len(df)
        all_features = []

        premises = list(df['premise'])
        hypotheses = list(df['hypothesis'])

        # Load spaCy if needed
        if self.use_spacy and self.nlp is None:
            from src.nli_spacy_features import get_spacy_model
            self.nlp = get_spacy_model()

        # Batch spaCy processing
        spacy_feats = None
        if self.use_spacy:
            from src.nli_spacy_features import batch_extract_spacy_features
            print("Computing spaCy features...")
            spacy_feats = batch_extract_spacy_features(
                premises, hypotheses, self.nlp
            )

        # Extract features for each pair
        iterator = range(n)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features")

        for i in iterator:
            premise = premises[i]
            hypothesis = hypotheses[i]

            pair_feats = {}

            # Basic features (Tiers 1, 3, 7)
            pair_feats.update(extract_basic_features(premise, hypothesis))

            # spaCy features (Tiers 4, 5, 6)
            if spacy_feats:
                pair_feats.update(spacy_feats[i])

            # TF-IDF similarity features (Tier 2)
            pair_feats.update(
                self.tfidf.compute_similarity_features(premise, hypothesis)
            )

            # Cross TF-IDF+SVD features (Tier 8)
            pair_feats.update(
                self.tfidf.compute_cross_features(premise, hypothesis)
            )

            # GloVe features
            if self.use_glove and self.glove:
                pair_feats.update(self.glove.compute_features(premise, hypothesis))

            # Interaction features (Tier 9)
            pair_feats.update(interaction_features(pair_feats))

            all_features.append(pair_feats)

        # Convert to numpy array
        if self._feature_names is None:
            self._feature_names = sorted(all_features[0].keys())

        X = np.array([
            [f.get(name, 0.0) for name in self._feature_names]
            for f in all_features
        ], dtype=np.float32)

        # Replace NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"Feature matrix shape: {X.shape}")
        return X, self._feature_names

    @property
    def feature_names(self):
        return self._feature_names
