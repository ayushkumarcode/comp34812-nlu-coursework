"""
AV Category A — Full feature extraction pipeline.
Combines all feature groups into a single feature matrix for training/inference.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.av_feature_engineering import (
    extract_per_text_features,
    pairwise_features,
    FUNCTION_WORDS,
)
from src.av_tfidf_features import CharNgramTFIDF, CosineSimFeatures


# Style-only feature prefixes (for topic-robustness mechanism)
STYLE_FEATURE_PREFIXES = [
    'fw_', 'pos_', 'avg_dep_depth', 'max_dep_depth', 'avg_branching_factor',
    'subordination_index', 'avg_dep_arc_length', 'passive_ratio', 'relcl_ratio',
    'avg_conjuncts', 'content_clause_ratio', 'fronted_adverb_ratio',
    'sent_len_autocorr', 'sent_len_entropy', 'punct_burstiness',
    'sent_len_var_ratio', 'sent_len_mean_reversion', 'punct_diversity_entropy',
    'char_bigram_mi', 'text_entropy_rate', 'char_cond_entropy',
    'word_length_entropy', 'rolling_ttr_entropy',
    'avg_sentence_length', 'std_sentence_length', 'median_sentence_length',
    'n_sentences', 'pct_short_sentences', 'pct_long_sentences',
    'exclamation_density', 'question_density', 'ellipsis_count_norm',
    'capitalization_ratio', 'quote_ratio',
]


class AVFeatureExtractor:
    """Complete AV Category A feature extraction pipeline.

    Handles:
    - Per-text features (Groups 1, 2, 4, 6, 8, 9)
    - spaCy features (Groups 5, 7) - optional, can be disabled
    - TF-IDF + SVD features (Group 3)
    - Cosine similarity features
    - Diff-vector computation
    - Pairwise features
    - Style-only diff-vector (topic-robustness mechanism)
    """

    def __init__(self, use_spacy=True, n_svd_components=100):
        self.use_spacy = use_spacy
        self.tfidf = CharNgramTFIDF(n_components=n_svd_components)
        self.cosine = CosineSimFeatures()
        self.nlp = None
        self._fitted = False
        self._feature_names = None

    def fit(self, df):
        """Fit TF-IDF vectorizers on training data.

        Args:
            df: Training DataFrame with text_1, text_2 columns.
        """
        all_texts = list(df['text_1']) + list(df['text_2'])
        print(f"Fitting TF-IDF on {len(all_texts)} texts...")
        self.tfidf.fit(all_texts)
        self.cosine.fit(all_texts)
        self._fitted = True
        return self

    def transform(self, df, show_progress=True):
        """Extract all features for a DataFrame of text pairs.

        Args:
            df: DataFrame with text_1, text_2 columns.
            show_progress: If True, show progress bar.

        Returns:
            numpy array of shape (n_pairs, n_features), list of feature names.
        """
        n = len(df)
        all_features = []

        # Load spaCy if needed
        if self.use_spacy and self.nlp is None:
            from src.av_spacy_features import get_spacy_model
            self.nlp = get_spacy_model()

        # Pre-compute TF-IDF SVD features for all texts
        print("Computing TF-IDF SVD features...")
        all_text_1 = list(df['text_1'])
        all_text_2 = list(df['text_2'])
        svd_1 = self.tfidf.transform(all_text_1)
        svd_2 = self.tfidf.transform(all_text_2)

        # Pre-compute spaCy features if needed
        spacy_feats_1 = None
        spacy_feats_2 = None
        if self.use_spacy:
            from src.av_spacy_features import batch_extract_spacy_features
            print("Computing spaCy features for text_1...")
            spacy_feats_1 = batch_extract_spacy_features(all_text_1, self.nlp)
            print("Computing spaCy features for text_2...")
            spacy_feats_2 = batch_extract_spacy_features(all_text_2, self.nlp)

        # Extract features for each pair
        iterator = range(n)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features")

        for i in iterator:
            text_1 = all_text_1[i]
            text_2 = all_text_2[i]

            pair_feats = {}

            # Per-text features (Groups 1, 2, 4, 6, 8, 9)
            feats_1 = extract_per_text_features(text_1)
            feats_2 = extract_per_text_features(text_2)

            # Add spaCy features (Groups 5, 7)
            if self.use_spacy:
                feats_1.update(spacy_feats_1[i])
                feats_2.update(spacy_feats_2[i])

            # Add TF-IDF SVD features (Group 3)
            for j in range(svd_1.shape[1]):
                feats_1[f'tfidf_svd_{j}'] = float(svd_1[i, j])
                feats_2[f'tfidf_svd_{j}'] = float(svd_2[i, j])

            # Full diff-vector: |f(text_1) - f(text_2)|
            for key in sorted(feats_1.keys()):
                pair_feats[f'diff_{key}'] = abs(feats_1[key] - feats_2[key])

            # Style-only diff-vector (topic-robustness)
            for key in sorted(feats_1.keys()):
                if any(key.startswith(prefix) for prefix in STYLE_FEATURE_PREFIXES):
                    pair_feats[f'style_diff_{key}'] = abs(feats_1[key] - feats_2[key])

            # Pairwise features
            pair_feats.update(pairwise_features(text_1, text_2))

            # Cosine similarity features
            pair_feats.update(self.cosine.compute_similarities(text_1, text_2))

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
