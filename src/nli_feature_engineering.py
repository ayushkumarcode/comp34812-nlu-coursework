"""
NLI Category A — Feature Engineering.
Comprehensive feature extraction for premise-hypothesis pairs.

Feature Tiers:
  1. Lexical Overlap (28)
  2. Semantic Similarity (18)
  3. Negation & Contradiction (16)
  4. Syntactic (20)
  5. Alignment-Based — Sultan et al. 2014 inspired (12)
  6. Natural Logic — MacCartney & Manning 2007/2009 inspired (8)
  7. Cross-Sentence Structural (10)
  8. BoW Cross Features — TF-IDF + SVD (100)
  9. Interaction Features (18)

Total: ~280 features per premise-hypothesis pair.
"""

import math
import re
from collections import Counter
from itertools import product as iter_product

import numpy as np


# ============================================================
# TIER 1: LEXICAL OVERLAP FEATURES (28 features)
# ============================================================

def lexical_overlap_features(premise, hypothesis):
    """Extract lexical overlap features between premise and hypothesis.

    Returns:
        Dict with ~28 overlap features.
    """
    feats = {}

    p_words = premise.lower().split()
    h_words = hypothesis.lower().split()

    p_set = set(p_words)
    h_set = set(h_words)

    # Unigram overlap ratios
    inter = p_set & h_set
    feats['unigram_overlap_p2h'] = len(inter) / max(len(h_set), 1)
    feats['unigram_overlap_h2p'] = len(inter) / max(len(p_set), 1)

    # Bigram overlap
    p_bigrams = set(zip(p_words[:-1], p_words[1:])) if len(p_words) > 1 else set()
    h_bigrams = set(zip(h_words[:-1], h_words[1:])) if len(h_words) > 1 else set()
    bg_inter = p_bigrams & h_bigrams
    feats['bigram_overlap_p2h'] = len(bg_inter) / max(len(h_bigrams), 1)
    feats['bigram_overlap_h2p'] = len(bg_inter) / max(len(p_bigrams), 1)

    # Trigram overlap
    p_trigrams = set(zip(p_words[:-2], p_words[1:-1], p_words[2:])) if len(p_words) > 2 else set()
    h_trigrams = set(zip(h_words[:-2], h_words[1:-1], h_words[2:])) if len(h_words) > 2 else set()
    tg_inter = p_trigrams & h_trigrams
    feats['trigram_overlap_p2h'] = len(tg_inter) / max(len(h_trigrams), 1)
    feats['trigram_overlap_h2p'] = len(tg_inter) / max(len(p_trigrams), 1)

    # Jaccard similarities
    union = p_set | h_set
    feats['jaccard_unigram'] = len(inter) / max(len(union), 1)
    bg_union = p_bigrams | h_bigrams
    feats['jaccard_bigram'] = len(bg_inter) / max(len(bg_union), 1)

    # BLEU scores (simplified: precision of n-grams)
    for n in [1, 2, 3, 4]:
        feats[f'bleu{n}_fwd'] = _bleu_n(p_words, h_words, n)
        feats[f'bleu{n}_rev'] = _bleu_n(h_words, p_words, n)

    # Length features
    feats['length_ratio'] = min(len(p_words) / max(len(h_words), 1), 10.0)
    feats['length_diff'] = abs(len(p_words) - len(h_words))

    # Content word coverage (exclude stopwords)
    stopwords = _STOPWORDS
    p_content = {w for w in p_set if w not in stopwords}
    h_content = {w for w in h_set if w not in stopwords}
    content_inter = p_content & h_content
    feats['hypothesis_content_coverage'] = len(content_inter) / max(len(h_content), 1)
    feats['premise_content_coverage'] = len(content_inter) / max(len(p_content), 1)

    # Exact match and substring containment
    feats['exact_match'] = 1.0 if premise.strip().lower() == hypothesis.strip().lower() else 0.0
    feats['substring_containment'] = 1.0 if hypothesis.lower() in premise.lower() else 0.0

    # Character-level trigram overlap (Jaccard)
    p_chars = premise.lower()
    h_chars = hypothesis.lower()
    p_char_tri = set(zip(p_chars[:-2], p_chars[1:-1], p_chars[2:])) if len(p_chars) > 2 else set()
    h_char_tri = set(zip(h_chars[:-2], h_chars[1:-1], h_chars[2:])) if len(h_chars) > 2 else set()
    ct_union = p_char_tri | h_char_tri
    ct_inter = p_char_tri & h_char_tri
    feats['char_trigram_jaccard'] = len(ct_inter) / max(len(ct_union), 1)

    # LCS ratio
    feats['lcs_ratio'] = _lcs_ratio(p_words, h_words)

    return feats


def _bleu_n(reference, candidate, n):
    """Simplified BLEU-n precision."""
    if len(candidate) < n or len(reference) < n:
        return 0.0
    ref_ngrams = Counter(zip(*[reference[i:] for i in range(n)]))
    cand_ngrams = Counter(zip(*[candidate[i:] for i in range(n)]))
    clipped = sum(min(cand_ngrams[ng], ref_ngrams.get(ng, 0)) for ng in cand_ngrams)
    total = sum(cand_ngrams.values())
    return clipped / max(total, 1)


def _lcs_ratio(seq1, seq2):
    """Longest common subsequence ratio."""
    if not seq1 or not seq2:
        return 0.0
    m, n = len(seq1), len(seq2)
    # Use space-optimized LCS
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev = curr
    lcs_len = prev[n]
    return lcs_len / max(m, n)


# ============================================================
# TIER 3: NEGATION & CONTRADICTION FEATURES (16 features)
# ============================================================

NEGATION_CUES = {
    'not', "n't", 'no', 'never', 'neither', 'nobody', 'nothing',
    'nowhere', 'hardly', 'scarcely', 'barely', 'seldom', 'nor',
    'cannot', "can't", "won't", "wouldn't", "shouldn't", "couldn't",
    "didn't", "doesn't", "don't", "isn't", "aren't", "wasn't", "weren't",
}

MODAL_VERBS = {
    'can', 'could', 'may', 'might', 'will', 'would', 'shall', 'should', 'must'
}

QUANTIFIERS = {
    'all', 'every', 'each', 'some', 'few', 'many', 'most', 'no',
    'none', 'any', 'several', 'both', 'either', 'neither'
}


def negation_contradiction_features(premise, hypothesis):
    """Extract negation and contradiction features.

    Returns:
        Dict with ~16 features.
    """
    feats = {}

    p_words = set(premise.lower().split())
    h_words = set(hypothesis.lower().split())
    p_tokens = premise.lower().split()
    h_tokens = hypothesis.lower().split()

    # Negation presence
    p_neg = p_words & NEGATION_CUES
    h_neg = h_words & NEGATION_CUES
    feats['neg_in_premise'] = 1.0 if p_neg else 0.0
    feats['neg_in_hypothesis'] = 1.0 if h_neg else 0.0
    feats['neg_mismatch'] = 1.0 if bool(p_neg) != bool(h_neg) else 0.0

    # Negation counts
    p_neg_count = sum(1 for w in p_tokens if w in NEGATION_CUES)
    h_neg_count = sum(1 for w in h_tokens if w in NEGATION_CUES)
    feats['neg_count_premise'] = p_neg_count
    feats['neg_count_hypothesis'] = h_neg_count
    feats['neg_count_diff'] = abs(p_neg_count - h_neg_count)

    # Number mismatch
    p_numbers = set(re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', premise))
    h_numbers = set(re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', hypothesis))
    feats['number_mismatch'] = 1.0 if p_numbers and h_numbers and not p_numbers & h_numbers else 0.0
    num_inter = p_numbers & h_numbers
    feats['number_overlap_ratio'] = len(num_inter) / max(len(h_numbers), 1) if h_numbers else 0.0

    # Modal verb mismatch
    p_modal = p_words & MODAL_VERBS
    h_modal = h_words & MODAL_VERBS
    feats['modal_mismatch'] = 1.0 if bool(p_modal) != bool(h_modal) else 0.0

    # Quantifier features
    p_quant = p_words & QUANTIFIERS
    h_quant = h_words & QUANTIFIERS
    feats['quantifier_mismatch'] = 1.0 if bool(p_quant) != bool(h_quant) else 0.0

    # Antonym placeholder (requires WordNet — handled in spaCy/WordNet features)
    feats['antonym_present'] = 0.0
    feats['antonym_count'] = 0.0

    # Sentiment polarity difference placeholder (simple heuristic)
    p_pos = sum(1 for w in p_tokens if w in _POSITIVE_WORDS)
    p_neg_sent = sum(1 for w in p_tokens if w in _NEGATIVE_WORDS)
    h_pos = sum(1 for w in h_tokens if w in _POSITIVE_WORDS)
    h_neg_sent = sum(1 for w in h_tokens if w in _NEGATIVE_WORDS)
    p_polarity = (p_pos - p_neg_sent) / max(len(p_tokens), 1)
    h_polarity = (h_pos - h_neg_sent) / max(len(h_tokens), 1)
    feats['sentiment_diff'] = abs(p_polarity - h_polarity)

    return feats


# ============================================================
# TIER 7: CROSS-SENTENCE STRUCTURAL FEATURES (10 features)
# ============================================================

def structural_features(premise, hypothesis):
    """Extract cross-sentence structural features.

    Returns:
        Dict with 10 features.
    """
    feats = {}

    p_sents = [s for s in re.split(r'(?<=[.!?])\s+', premise.strip()) if s.strip()]
    h_sents = [s for s in re.split(r'(?<=[.!?])\s+', hypothesis.strip()) if s.strip()]

    feats['premise_sent_count'] = len(p_sents)
    feats['hypothesis_sent_count'] = len(h_sents)
    feats['sent_count_ratio'] = len(p_sents) / max(len(h_sents), 1)

    p_words = premise.split()
    h_words = hypothesis.split()

    p_avg_wl = np.mean([len(w) for w in p_words]) if p_words else 0.0
    h_avg_wl = np.mean([len(w) for w in h_words]) if h_words else 0.0
    feats['avg_word_len_premise'] = p_avg_wl
    feats['avg_word_len_hypothesis'] = h_avg_wl
    feats['word_len_ratio'] = p_avg_wl / max(h_avg_wl, 1e-8)

    feats['premise_is_question'] = 1.0 if premise.rstrip().endswith('?') else 0.0
    feats['hypothesis_is_question'] = 1.0 if hypothesis.rstrip().endswith('?') else 0.0

    feats['premise_word_count'] = len(p_words)
    feats['hypothesis_word_count'] = len(h_words)

    return feats


# ============================================================
# TIER 9: INTERACTION FEATURES (18 features)
# ============================================================

def interaction_features(base_feats):
    """Compute interaction features from base features.

    Args:
        base_feats: Dict of already-computed features.

    Returns:
        Dict with ~18 interaction features.
    """
    feats = {}

    def _get(name, default=0.0):
        return base_feats.get(name, default)

    feats['interact_overlap_x_negmismatch'] = _get('unigram_overlap_p2h') * _get('neg_mismatch')
    feats['interact_coverage_x_antonym'] = _get('hypothesis_content_coverage') * _get('antonym_present')
    feats['interact_jaccard_x_lenratio'] = _get('jaccard_unigram') * _get('length_ratio')
    feats['interact_tfidf_x_negmismatch'] = _get('tfidf_cosine') * _get('neg_mismatch')
    feats['interact_coverage_x_contradiction'] = _get('hypothesis_content_coverage') * _get('contradiction_score')
    feats['interact_coverage_x_entailment'] = _get('hypothesis_content_coverage') * _get('entailment_score')
    feats['interact_wordnet_x_coverage'] = _get('wordnet_wup_max') * _get('hypothesis_content_coverage')
    feats['interact_lsa_x_negdiff'] = _get('lsa_cosine') * _get('neg_count_diff')
    feats['interact_sif_x_antonym'] = _get('sif_cosine') * _get('antonym_present')
    feats['interact_bleu4_x_negmismatch'] = _get('bleu4_fwd') * _get('neg_mismatch')
    feats['interact_nummismatch_x_tfidf'] = _get('number_mismatch') * _get('tfidf_cosine')
    feats['interact_exact_x_lenratio'] = _get('exact_match') * _get('length_ratio')
    feats['interact_substring_x_hwordcount'] = _get('substring_containment') * _get('hypothesis_word_count')
    feats['interact_modal_x_tfidf'] = _get('modal_mismatch') * _get('tfidf_cosine')
    feats['interact_quant_x_overlap'] = _get('quantifier_mismatch') * _get('unigram_overlap_p2h')
    feats['interact_sentiment_x_alignment'] = _get('sentiment_diff') * _get('alignment_symmetry')
    feats['interact_rootverb_x_coverage'] = _get('root_verb_match') * _get('hypothesis_content_coverage')
    feats['interact_lcsratio_x_negmismatch'] = _get('lcs_ratio') * _get('neg_mismatch')

    return feats


# ============================================================
# COMBINED EXTRACTION (TIERS 1, 3, 7, 9)
# ============================================================

def extract_basic_features(premise, hypothesis):
    """Extract all non-spaCy, non-WordNet, non-TF-IDF features.

    Combines Tiers 1, 3 (negation), 7 (structural).
    Tier 9 (interactions) added after all features are assembled.

    Returns:
        Dict of feature name -> float.
    """
    feats = {}
    feats.update(lexical_overlap_features(premise, hypothesis))
    feats.update(negation_contradiction_features(premise, hypothesis))
    feats.update(structural_features(premise, hypothesis))
    return feats


# ============================================================
# UTILITY: STOPWORDS AND SENTIMENT WORDS
# ============================================================

_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'and', 'but', 'or', 'nor',
    'not', 'so', 'than', 'too', 'very', 'just', 'about', 'up', 'its', 'it',
    'he', 'she', 'they', 'we', 'you', 'i', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'our', 'their', 'this', 'that', 'these', 'those',
}

_POSITIVE_WORDS = {
    'good', 'great', 'best', 'better', 'happy', 'love', 'like', 'wonderful',
    'excellent', 'amazing', 'beautiful', 'nice', 'right', 'correct', 'true',
    'positive', 'success', 'win', 'won', 'perfect', 'fine', 'well',
}

_NEGATIVE_WORDS = {
    'bad', 'worst', 'worse', 'sad', 'hate', 'terrible', 'awful', 'wrong',
    'false', 'negative', 'fail', 'failed', 'poor', 'ugly', 'horrible',
    'loss', 'lost', 'broke', 'broken', 'dead', 'die', 'died', 'kill',
}
