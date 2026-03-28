"""
AV Cat A feature engineering -- the stylometric feature extraction.
9 feature groups total, ~695 features per text pair when combined with
pairwise measures and diff-vectors. Groups 7-9 are our novel contributions.
"""

import math
import gzip
import lzma
import bz2
import re
from collections import Counter

import numpy as np


# -- group 1: lexical features (~30) --

def lexical_features(text):
    """Extract lexical features like TTR, Yule's K, hapax ratios, etc."""
    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return _empty_lexical()

    feats = {}

    word_lengths = [len(w) for w in words]
    feats['avg_word_length'] = np.mean(word_lengths)

    length_counts = Counter(min(l, 20) for l in word_lengths)
    for i in range(1, 21):
        feats[f'word_len_prop_{i}'] = length_counts.get(i, 0) / n_words

    word_lower = [w.lower() for w in words]
    freq = Counter(word_lower)
    vocab_size = len(freq)

    feats['ttr'] = vocab_size / n_words if n_words > 0 else 0

    hapax = sum(1 for c in freq.values() if c == 1)
    feats['hapax_ratio'] = hapax / n_words if n_words > 0 else 0

    hapax_dis = sum(1 for c in freq.values() if c == 2)
    feats['hapax_dis_ratio'] = hapax_dis / n_words if n_words > 0 else 0

    if n_words < 50:
        # too few words for reliable richness measures
        feats['yules_k'] = 0.0
        feats['simpsons_d'] = 0.0
        feats['honores_r'] = 0.0
        feats['brunets_w'] = 0.0
    else:
        freq_spectrum = Counter(freq.values())
        sum_fi2 = sum(i * i * vi for i, vi in freq_spectrum.items())
        if n_words > 1:
            feats['yules_k'] = 1e4 * (sum_fi2 - n_words) / (n_words * n_words)
        else:
            feats['yules_k'] = 0.0

        n_pairs = n_words * (n_words - 1)
        if n_pairs > 0:
            sum_ni = sum(c * (c - 1) for c in freq.values())
            feats['simpsons_d'] = 1 - sum_ni / n_pairs
        else:
            feats['simpsons_d'] = 0.0

        if hapax < vocab_size and vocab_size > 0 and n_words > 0:
            feats['honores_r'] = 100 * math.log(n_words) / (1 - hapax / vocab_size)
        else:
            feats['honores_r'] = 0.0

        if vocab_size > 0 and n_words > 0:
            feats['brunets_w'] = n_words ** (vocab_size ** -0.172)
        else:
            feats['brunets_w'] = 0.0

    feats['word_count'] = n_words

    return feats


def _empty_lexical():
    """Return zeroed lexical features for empty text."""
    feats = {'avg_word_length': 0.0}
    for i in range(1, 21):
        feats[f'word_len_prop_{i}'] = 0.0
    for name in ['ttr', 'hapax_ratio', 'hapax_dis_ratio', 'yules_k',
                  'simpsons_d', 'honores_r', 'brunets_w', 'word_count']:
        feats[name] = 0.0
    return feats


# -- group 2: character-level features (56) --

_SPECIAL_CHARS = list('.,;:!?\'"()-/\\@#$%&*_')


def character_features(text):
    """Character frequency features: 26 letters + 10 digits + 20 special chars."""
    feats = {}
    total_chars = len(text)
    if total_chars == 0:
        for c in 'abcdefghijklmnopqrstuvwxyz':
            feats[f'char_freq_{c}'] = 0.0
        for d in range(10):
            feats[f'digit_freq_{d}'] = 0.0
        for sc in _SPECIAL_CHARS:
            feats[f'special_freq_{ord(sc)}'] = 0.0
        return feats

    text_lower = text.lower()
    char_counts = Counter(text_lower)

    # Letter frequencies (a-z)
    for c in 'abcdefghijklmnopqrstuvwxyz':
        feats[f'char_freq_{c}'] = char_counts.get(c, 0) / total_chars

    # Digit frequencies (0-9)
    digit_counts = Counter(c for c in text if c.isdigit())
    for d in range(10):
        feats[f'digit_freq_{d}'] = digit_counts.get(str(d), 0) / total_chars

    # Special character frequencies
    for sc in _SPECIAL_CHARS:
        feats[f'special_freq_{ord(sc)}'] = text.count(sc) / total_chars

    return feats


# -- group 4: function word frequencies (150) --

FUNCTION_WORDS = [
    'the', 'of', 'and', 'a', 'to', 'in', 'is', 'it', 'that', 'was',
    'for', 'on', 'are', 'with', 'as', 'i', 'his', 'they', 'be', 'at',
    'one', 'have', 'this', 'from', 'or', 'had', 'by', 'but', 'not', 'what',
    'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'each', 'which',
    'she', 'do', 'how', 'their', 'if', 'will', 'up', 'about', 'out', 'many',
    'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into',
    'him', 'has', 'two', 'more', 'no', 'way', 'could', 'my', 'than', 'first',
    'been', 'who', 'its', 'now', 'over', 'just', 'other', 'also', 'after', 'very',
    'because', 'before', 'however', 'most', 'should', 'where', 'still', 'must',
    'while', 'therefore', 'although', 'since', 'during', 'until', 'unless',
    'instead', 'neither', 'either', 'moreover', 'furthermore', 'consequently',
    'nevertheless', 'thus', 'hence', 'accordingly', 'yet', 'though', 'whether',
    'per', 'both', 'such', 'those', 'any', 'own', 'an', 'only', 'being', 'did',
    'another', 'may', 'might', 'shall', 'upon', 'much', 'often', 'perhaps',
    'again', 'too', 'once', 'already', 'above', 'below', 'between', 'through',
    'around', 'against', 'without', 'within', 'along', 'among', 'across',
    'toward', 'including', 'following', 'according', 'regarding',
    'concerning', 'despite', 'except', 'beyond', 'besides', 'underneath',
    'alongside', 'outside', 'inside', 'throughout', 'beneath', 'behind',
    'ahead', 'apart', 'aside', 'away', 'everywhere', 'nowhere', 'somehow',
    'somewhat', 'sometimes', 'somewhere', 'otherwise',
]


def function_word_features(text):
    """Frequency of each function word, normalized by total word count."""
    words = text.lower().split()
    n_words = len(words)
    word_counts = Counter(words)

    feats = {}
    for fw in FUNCTION_WORDS:
        feats[f'fw_{fw}'] = word_counts.get(fw, 0) / n_words if n_words > 0 else 0.0
    return feats


# -- group 6: structural features (15) --

def structural_features(text):
    """Sentence lengths, paragraph counts, punctuation densities, etc."""
    feats = {}

    # Sentence splitting (simple heuristic: split on .!? followed by space or end)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    n_sentences = max(len(sentences), 1)

    sent_lengths = [len(s.split()) for s in sentences]

    feats['avg_sentence_length'] = np.mean(sent_lengths) if sent_lengths else 0.0
    feats['std_sentence_length'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0.0
    feats['median_sentence_length'] = np.median(sent_lengths) if sent_lengths else 0.0
    feats['max_sentence_length'] = max(sent_lengths) if sent_lengths else 0.0
    feats['min_sentence_length'] = min(sent_lengths) if sent_lengths else 0.0
    feats['n_sentences'] = n_sentences

    feats['pct_short_sentences'] = sum(1 for l in sent_lengths if l < 8) / n_sentences
    feats['pct_long_sentences'] = sum(1 for l in sent_lengths if l > 25) / n_sentences

    # Paragraph features
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    n_paragraphs = max(len(paragraphs), 1)
    feats['avg_words_per_paragraph'] = len(text.split()) / n_paragraphs
    feats['n_paragraphs'] = n_paragraphs

    # Quote ratio
    quote_chars = text.count('"') + text.count("'") + text.count('\u201c') + text.count('\u201d')
    feats['quote_ratio'] = quote_chars / len(text) if len(text) > 0 else 0.0

    # Punctuation densities
    feats['exclamation_density'] = text.count('!') / n_sentences
    feats['question_density'] = text.count('?') / n_sentences
    feats['ellipsis_count_norm'] = text.count('...') / n_sentences

    # Capitalization ratio
    letters = [c for c in text if c.isalpha()]
    if letters:
        feats['capitalization_ratio'] = sum(1 for c in letters if c.isupper()) / len(letters)
    else:
        feats['capitalization_ratio'] = 0.0

    return feats


# -- group 8: writing rhythm features (6) -- NOVEL

def writing_rhythm_features(text):
    """Writing rhythm: autocorrelation, burstiness, mean-reversion patterns."""
    feats = {}

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    sent_lengths = [len(s.split()) for s in sentences]

    if len(sent_lengths) < 3:
        return {
            'sent_len_autocorr': 0.0,
            'sent_len_entropy': 0.0,
            'punct_burstiness': 0.0,
            'sent_len_var_ratio': 0.0,
            'sent_len_mean_reversion': 0.0,
            'punct_diversity_entropy': 0.0,
        }

    # Sentence length autocorrelation (lag-1)
    sl = np.array(sent_lengths, dtype=float)
    sl_mean = sl.mean()
    sl_var = sl.var()
    if sl_var > 0:
        autocorr = np.corrcoef(sl[:-1], sl[1:])[0, 1]
        feats['sent_len_autocorr'] = autocorr if not np.isnan(autocorr) else 0.0
    else:
        feats['sent_len_autocorr'] = 0.0

    # Entropy of sentence length distribution
    bins = [0, 5, 10, 15, 20, 30, 50, 200]
    hist, _ = np.histogram(sent_lengths, bins=bins)
    probs = hist / hist.sum() if hist.sum() > 0 else hist
    probs = probs[probs > 0]
    feats['sent_len_entropy'] = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0

    # Burstiness of punctuation (CV of inter-punctuation distances)
    punct_positions = [i for i, c in enumerate(text) if c in '.,;:!?']
    if len(punct_positions) > 2:
        distances = np.diff(punct_positions).astype(float)
        mean_d = distances.mean()
        if mean_d > 0:
            feats['punct_burstiness'] = distances.std() / mean_d
        else:
            feats['punct_burstiness'] = 0.0
    else:
        feats['punct_burstiness'] = 0.0

    # Sentence length variance ratio (first half vs second half)
    mid = len(sent_lengths) // 2
    if mid > 0:
        var_first = np.var(sent_lengths[:mid])
        var_second = np.var(sent_lengths[mid:])
        feats['sent_len_var_ratio'] = var_first / (var_second + 1e-8)
    else:
        feats['sent_len_var_ratio'] = 1.0

    # Mean-reversion tendency
    if len(sent_lengths) > 2 and sl_var > 0:
        deviations = sl - sl_mean
        reversion = -np.corrcoef(deviations[:-1], np.diff(sl))[0, 1]
        feats['sent_len_mean_reversion'] = reversion if not np.isnan(reversion) else 0.0
    else:
        feats['sent_len_mean_reversion'] = 0.0

    # Punctuation diversity entropy
    punct_types = [c for c in text if c in '.,;:!?-\'"()']
    if punct_types:
        punct_counts = Counter(punct_types)
        total = sum(punct_counts.values())
        probs = np.array([c / total for c in punct_counts.values()])
        feats['punct_diversity_entropy'] = -np.sum(probs * np.log2(probs))
    else:
        feats['punct_diversity_entropy'] = 0.0

    return feats


# -- group 9: information-theoretic features (5) -- NOVEL

def info_theoretic_features(text):
    """Info-theoretic features: char bigram MI, entropy rate, conditional entropy."""
    feats = {}

    if len(text) < 10:
        return {
            'char_bigram_mi': 0.0,
            'text_entropy_rate': 0.0,
            'char_cond_entropy': 0.0,
            'word_length_entropy': 0.0,
            'rolling_ttr_entropy': 0.0,
        }

    # Character bigram mutual information
    chars = list(text.lower())
    n = len(chars)
    unigram_counts = Counter(chars)
    bigram_counts = Counter(zip(chars[:-1], chars[1:]))
    n_bigrams = n - 1

    if n_bigrams > 0:
        pmi_values = []
        for (c1, c2), count in bigram_counts.items():
            p_bigram = count / n_bigrams
            p_c1 = unigram_counts[c1] / n
            p_c2 = unigram_counts[c2] / n
            if p_c1 > 0 and p_c2 > 0 and p_bigram > 0:
                pmi_values.append(math.log2(p_bigram / (p_c1 * p_c2)))
        feats['char_bigram_mi'] = np.mean(pmi_values) if pmi_values else 0.0
    else:
        feats['char_bigram_mi'] = 0.0

    # Text entropy rate (bits per character)
    probs = np.array([c / n for c in unigram_counts.values()])
    feats['text_entropy_rate'] = -np.sum(probs * np.log2(probs))

    # Conditional entropy H(c_n | c_{n-1})
    if n_bigrams > 0:
        h_bigram = 0.0
        for count in bigram_counts.values():
            p = count / n_bigrams
            if p > 0:
                h_bigram -= p * math.log2(p)
        h_unigram = feats['text_entropy_rate']
        feats['char_cond_entropy'] = h_bigram - h_unigram
    else:
        feats['char_cond_entropy'] = 0.0

    # Word length entropy
    words = text.split()
    if words:
        word_lens = [len(w) for w in words]
        len_counts = Counter(word_lens)
        total = sum(len_counts.values())
        probs = np.array([c / total for c in len_counts.values()])
        feats['word_length_entropy'] = -np.sum(probs * np.log2(probs))
    else:
        feats['word_length_entropy'] = 0.0

    # Rolling TTR entropy (50-word windows)
    if len(words) >= 50:
        ttrs = []
        for i in range(0, len(words) - 49):
            window = words[i:i + 50]
            ttrs.append(len(set(w.lower() for w in window)) / 50)
        if len(ttrs) > 1:
            bins = np.linspace(0, 1, 11)
            hist, _ = np.histogram(ttrs, bins=bins)
            total = hist.sum()
            if total > 0:
                probs = hist[hist > 0] / total
                feats['rolling_ttr_entropy'] = -np.sum(probs * np.log2(probs))
            else:
                feats['rolling_ttr_entropy'] = 0.0
        else:
            feats['rolling_ttr_entropy'] = 0.0
    else:
        feats['rolling_ttr_entropy'] = 0.0

    return feats


# -- pairwise features (14) --

def pairwise_features(text_1, text_2):
    """Pairwise similarity/distance features between two texts (~14 features)."""
    feats = {}

    # NCD with gzip
    feats['ncd_gzip'] = _ncd(text_1, text_2, gzip.compress)

    # NCD with lzma
    feats['ncd_lzma'] = _ncd(text_1, text_2, lzma.compress)

    # NCD with bz2
    feats['ncd_bz2'] = _ncd(text_1, text_2, bz2.compress)

    # Word overlap features
    words_1 = set(text_1.lower().split())
    words_2 = set(text_2.lower().split())

    # Word overlap (Jaccard)
    union = words_1 | words_2
    intersection = words_1 & words_2
    feats['word_jaccard'] = len(intersection) / len(union) if union else 0.0

    # Content word overlap (non-function words)
    fw_set = set(FUNCTION_WORDS)
    cw_1 = words_1 - fw_set
    cw_2 = words_2 - fw_set
    cw_union = cw_1 | cw_2
    cw_intersection = cw_1 & cw_2
    feats['content_word_overlap'] = len(cw_intersection) / len(cw_union) if cw_union else 0.0

    # Length ratio
    len_1 = len(text_1)
    len_2 = len(text_2)
    feats['length_ratio'] = min(len_1, len_2) / max(len_1, len_2) if max(len_1, len_2) > 0 else 1.0

    # Word count ratio
    wc_1 = len(text_1.split())
    wc_2 = len(text_2.split())
    feats['word_count_ratio'] = min(wc_1, wc_2) / max(wc_1, wc_2) if max(wc_1, wc_2) > 0 else 1.0

    # Vocabulary size ratio
    v_1 = len(words_1)
    v_2 = len(words_2)
    feats['vocab_size_ratio'] = min(v_1, v_2) / max(v_1, v_2) if max(v_1, v_2) > 0 else 1.0

    # Jensen-Shannon divergence of word frequency distributions
    feats['jsd_word_freq'] = _jsd_word_freq(text_1, text_2)

    # Jensen-Shannon divergence of character bigram distributions
    feats['jsd_char_bigram'] = _jsd_char_bigram(text_1, text_2)

    # Burrows' Delta
    feats['burrows_delta'] = _burrows_delta(text_1, text_2)

    return feats


def _ncd(text_1, text_2, compress_fn):
    """NCD between two texts using the given compressor."""
    b1 = text_1.encode('utf-8')
    b2 = text_2.encode('utf-8')
    c1 = len(compress_fn(b1))
    c2 = len(compress_fn(b2))
    c12 = len(compress_fn(b1 + b2))
    return (c12 - min(c1, c2)) / max(c1, c2) if max(c1, c2) > 0 else 0.0


def _jsd_word_freq(text_1, text_2):
    """JSD between word frequency distributions of two texts."""
    words_1 = text_1.lower().split()
    words_2 = text_2.lower().split()
    if not words_1 or not words_2:
        return 0.0

    freq_1 = Counter(words_1)
    freq_2 = Counter(words_2)

    all_words = set(freq_1.keys()) | set(freq_2.keys())
    n1 = sum(freq_1.values())
    n2 = sum(freq_2.values())

    p = np.array([freq_1.get(w, 0) / n1 for w in all_words])
    q = np.array([freq_2.get(w, 0) / n2 for w in all_words])

    return _jsd(p, q)


def _jsd_char_bigram(text_1, text_2):
    """JSD between char bigram distributions."""
    bg_1 = Counter(zip(text_1.lower()[:-1], text_1.lower()[1:]))
    bg_2 = Counter(zip(text_2.lower()[:-1], text_2.lower()[1:]))
    if not bg_1 or not bg_2:
        return 0.0

    all_bigrams = set(bg_1.keys()) | set(bg_2.keys())
    n1 = sum(bg_1.values())
    n2 = sum(bg_2.values())

    p = np.array([bg_1.get(b, 0) / n1 for b in all_bigrams])
    q = np.array([bg_2.get(b, 0) / n2 for b in all_bigrams])

    return _jsd(p, q)


def _jsd(p, q):
    """JSD between two probability vectors."""
    m = 0.5 * (p + q)
    eps = 1e-10
    kl_pm = np.sum(p * np.log2((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log2((q + eps) / (m + eps)))
    return 0.5 * (kl_pm + kl_qm)


def _burrows_delta(text_1, text_2, n_top=100):
    """Burrows' Delta -- only meaningful when both texts have 200+ words."""
    words_1 = text_1.lower().split()
    words_2 = text_2.lower().split()

    if len(words_1) < 200 or len(words_2) < 200:
        return 0.0  # gets imputed later

    freq_1 = Counter(words_1)
    freq_2 = Counter(words_2)
    n1 = len(words_1)
    n2 = len(words_2)

    # Use combined most frequent words
    combined = Counter(words_1 + words_2)
    top_words = [w for w, _ in combined.most_common(n_top)]

    # Z-score normalization using combined corpus stats
    all_words = words_1 + words_2
    combined_freq = Counter(all_words)
    n_total = len(all_words)

    delta = 0.0
    for word in top_words:
        f1 = freq_1.get(word, 0) / n1
        f2 = freq_2.get(word, 0) / n2
        mean_f = combined_freq.get(word, 0) / n_total
        # Standard deviation across both texts
        std_f = max(abs(f1 - mean_f), abs(f2 - mean_f))
        if std_f > 0:
            delta += abs((f1 - mean_f) / std_f - (f2 - mean_f) / std_f)

    return delta / len(top_words) if top_words else 0.0


# -- combined feature extraction --

def extract_per_text_features(text):
    """All per-text features (groups 1,2,4,6,8,9). TF-IDF and spaCy are separate."""
    feats = {}
    feats.update(lexical_features(text))
    feats.update(character_features(text))
    feats.update(function_word_features(text))
    feats.update(structural_features(text))
    feats.update(writing_rhythm_features(text))
    feats.update(info_theoretic_features(text))
    return feats


def extract_pair_features(text_1, text_2, include_per_text=True):
    """Diff-vector + pairwise features for a text pair."""
    feats = {}

    if include_per_text:
        feats_1 = extract_per_text_features(text_1)
        feats_2 = extract_per_text_features(text_2)

        for key in feats_1:
            feats[f'diff_{key}'] = abs(feats_1[key] - feats_2[key])

    pair_feats = pairwise_features(text_1, text_2)
    feats.update(pair_feats)

    return feats
