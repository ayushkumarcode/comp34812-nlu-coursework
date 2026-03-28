"""
AV Cat A feature engineering -- the stylometric feature extraction.
13 feature groups total, ~715 features per text pair when combined with
pairwise measures and diff-vectors. Groups 7-13 are our novel contributions.
"""

import math
import gzip
import lzma
import bz2
import re
from collections import Counter

import numpy as np
from scipy.optimize import curve_fit


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

    for c in 'abcdefghijklmnopqrstuvwxyz':
        feats[f'char_freq_{c}'] = char_counts.get(c, 0) / total_chars

    digit_counts = Counter(c for c in text if c.isdigit())
    for d in range(10):
        feats[f'digit_freq_{d}'] = digit_counts.get(str(d), 0) / total_chars

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

    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    n_paragraphs = max(len(paragraphs), 1)
    feats['avg_words_per_paragraph'] = len(text.split()) / n_paragraphs
    feats['n_paragraphs'] = n_paragraphs

    quote_chars = text.count('"') + text.count("'") + text.count('\u201c') + text.count('\u201d')
    feats['quote_ratio'] = quote_chars / len(text) if len(text) > 0 else 0.0

    feats['exclamation_density'] = text.count('!') / n_sentences
    feats['question_density'] = text.count('?') / n_sentences
    feats['ellipsis_count_norm'] = text.count('...') / n_sentences

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

    sl = np.array(sent_lengths, dtype=float)
    sl_mean = sl.mean()
    sl_var = sl.var()
    if sl_var > 0:
        autocorr = np.corrcoef(sl[:-1], sl[1:])[0, 1]
        feats['sent_len_autocorr'] = autocorr if not np.isnan(autocorr) else 0.0
    else:
        feats['sent_len_autocorr'] = 0.0

    bins = [0, 5, 10, 15, 20, 30, 50, 200]
    hist, _ = np.histogram(sent_lengths, bins=bins)
    probs = hist / hist.sum() if hist.sum() > 0 else hist
    probs = probs[probs > 0]
    feats['sent_len_entropy'] = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0

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

    mid = len(sent_lengths) // 2
    if mid > 0:
        var_first = np.var(sent_lengths[:mid])
        var_second = np.var(sent_lengths[mid:])
        feats['sent_len_var_ratio'] = var_first / (var_second + 1e-8)
    else:
        feats['sent_len_var_ratio'] = 1.0

    if len(sent_lengths) > 2 and sl_var > 0:
        deviations = sl - sl_mean
        reversion = -np.corrcoef(deviations[:-1], np.diff(sl))[0, 1]
        feats['sent_len_mean_reversion'] = reversion if not np.isnan(reversion) else 0.0
    else:
        feats['sent_len_mean_reversion'] = 0.0

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

    probs = np.array([c / n for c in unigram_counts.values()])
    feats['text_entropy_rate'] = -np.sum(probs * np.log2(probs))

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

    words = text.split()
    if words:
        word_lens = [len(w) for w in words]
        len_counts = Counter(word_lens)
        total = sum(len_counts.values())
        probs = np.array([c / total for c in len_counts.values()])
        feats['word_length_entropy'] = -np.sum(probs * np.log2(probs))
    else:
        feats['word_length_entropy'] = 0.0

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


# -- group 10: FFT spectral analysis of sentence lengths (8) -- NOVEL

def spectral_features(text):
    """FFT of sentence-length series: dominant freq, centroid, band energies."""
    feats = {}
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    sent_lengths = np.array([len(s.split()) for s in sentences], dtype=float)

    if len(sent_lengths) < 4:
        return {
            'fft_dominant_freq': 0.0,
            'fft_spectral_centroid': 0.0,
            'fft_spectral_entropy': 0.0,
            'fft_energy_low': 0.0,
            'fft_energy_mid': 0.0,
            'fft_energy_high': 0.0,
            'fft_peak_to_avg': 0.0,
            'fft_spectral_rolloff': 0.0,
        }

    # zero-mean before FFT
    sl = sent_lengths - sent_lengths.mean()
    spectrum = np.abs(np.fft.rfft(sl))
    freqs = np.fft.rfftfreq(len(sl))

    # skip DC component (index 0)
    spectrum = spectrum[1:]
    freqs = freqs[1:]

    if len(spectrum) == 0 or spectrum.sum() == 0:
        return {k: 0.0 for k in [
            'fft_dominant_freq', 'fft_spectral_centroid',
            'fft_spectral_entropy', 'fft_energy_low',
            'fft_energy_mid', 'fft_energy_high',
            'fft_peak_to_avg', 'fft_spectral_rolloff',
        ]}

    total_energy = spectrum.sum()
    feats['fft_dominant_freq'] = freqs[np.argmax(spectrum)]
    feats['fft_spectral_centroid'] = np.sum(freqs * spectrum) / total_energy

    # spectral entropy
    probs = spectrum / total_energy
    probs = probs[probs > 0]
    feats['fft_spectral_entropy'] = -np.sum(probs * np.log2(probs))

    # band energies (low/mid/high thirds)
    n_bins = len(spectrum)
    third = max(n_bins // 3, 1)
    feats['fft_energy_low'] = spectrum[:third].sum() / total_energy
    feats['fft_energy_mid'] = spectrum[third:2*third].sum() / total_energy
    feats['fft_energy_high'] = spectrum[2*third:].sum() / total_energy

    feats['fft_peak_to_avg'] = spectrum.max() / (total_energy / n_bins)

    # spectral rolloff (freq below which 85% of energy lives)
    cumulative = np.cumsum(spectrum)
    rolloff_idx = np.searchsorted(cumulative, 0.85 * total_energy)
    rolloff_idx = min(rolloff_idx, len(freqs) - 1)
    feats['fft_spectral_rolloff'] = freqs[rolloff_idx]

    return feats


# -- group 11: Zipf-Mandelbrot law deviation (5) -- NOVEL

def zipf_features(text):
    """Fit Zipf-Mandelbrot law to word freq distribution, measure deviation."""
    feats = {}
    words = text.lower().split()

    if len(words) < 20:
        return {
            'zipf_alpha': 0.0,
            'zipf_beta': 0.0,
            'zipf_gof_residual': 0.0,
            'zipf_kl_div': 0.0,
            'zipf_r_squared': 0.0,
        }

    freq = Counter(words)
    counts = sorted(freq.values(), reverse=True)
    ranks = np.arange(1, len(counts) + 1, dtype=float)
    observed = np.array(counts, dtype=float)
    observed_norm = observed / observed.sum()

    # Zipf-Mandelbrot: f(r) = C / (r + beta)^alpha
    def zipf_mandelbrot(r, alpha, beta, c):
        return c / (r + beta) ** alpha

    try:
        popt, _ = curve_fit(
            zipf_mandelbrot, ranks, observed,
            p0=[1.0, 0.0, observed[0]],
            bounds=([0.01, -0.5, 0.01], [5.0, 50.0, observed[0] * 10]),
            maxfev=2000,
        )
        alpha, beta, c = popt
    except (RuntimeError, ValueError):
        alpha, beta, c = 1.0, 0.0, observed[0]

    feats['zipf_alpha'] = alpha
    feats['zipf_beta'] = beta

    # goodness of fit (normalized chi-squared residual)
    expected = zipf_mandelbrot(ranks, alpha, beta, c)
    expected_norm = expected / expected.sum()
    residuals = (observed_norm - expected_norm) ** 2
    feats['zipf_gof_residual'] = residuals.sum()

    # KL divergence from fitted distribution
    eps = 1e-10
    feats['zipf_kl_div'] = np.sum(
        observed_norm * np.log((observed_norm + eps) / (expected_norm + eps))
    )

    # R-squared of the fit
    ss_res = np.sum((observed - expected) ** 2)
    ss_tot = np.sum((observed - observed.mean()) ** 2)
    feats['zipf_r_squared'] = 1.0 - ss_res / (ss_tot + eps)

    return feats


# -- group 12: Benford's law on linguistic distributions (4) -- NOVEL

def benford_features(text):
    """Check first-digit distribution of word freq ranks vs Benford's law."""
    feats = {}
    words = text.lower().split()

    if len(words) < 20:
        return {
            'benford_chi2': 0.0,
            'benford_kl_div': 0.0,
            'benford_correlation': 0.0,
            'benford_mad': 0.0,
        }

    # Benford's theoretical distribution for digits 1-9
    benford_probs = np.array([
        math.log10(1 + 1.0/d) for d in range(1, 10)
    ])

    freq = Counter(words)
    counts = sorted(freq.values(), reverse=True)

    # first digits of word frequency counts (only multi-digit)
    first_digits = []
    for c in counts:
        if c >= 1:
            first_digits.append(int(str(c)[0]))

    if len(first_digits) < 10:
        return {
            'benford_chi2': 0.0,
            'benford_kl_div': 0.0,
            'benford_correlation': 0.0,
            'benford_mad': 0.0,
        }

    digit_counts = Counter(first_digits)
    total = sum(digit_counts.values())
    observed = np.array([digit_counts.get(d, 0) / total for d in range(1, 10)])

    # chi-squared statistic
    expected = benford_probs
    chi2 = np.sum((observed - expected) ** 2 / (expected + 1e-10))
    feats['benford_chi2'] = chi2

    # KL divergence
    eps = 1e-10
    feats['benford_kl_div'] = np.sum(
        observed * np.log((observed + eps) / (expected + eps))
    )

    # correlation with Benford
    if np.std(observed) > 0:
        feats['benford_correlation'] = np.corrcoef(observed, expected)[0, 1]
    else:
        feats['benford_correlation'] = 0.0

    # mean absolute deviation
    feats['benford_mad'] = np.mean(np.abs(observed - expected))

    return feats


# -- group 13: Hurst exponent / fractal analysis (3) -- NOVEL

def hurst_features(text):
    """Hurst exponent via R/S analysis on sentence-length series."""
    feats = {}
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    sent_lengths = np.array([len(s.split()) for s in sentences], dtype=float)

    if len(sent_lengths) < 8:
        return {
            'hurst_exponent': 0.5,
            'hurst_rs_intercept': 0.0,
            'hurst_stability': 0.0,
        }

    n = len(sent_lengths)

    # R/S analysis across multiple window sizes
    # use sizes from 4 up to n/2
    min_window = 4
    max_window = n // 2
    if max_window < min_window:
        return {
            'hurst_exponent': 0.5,
            'hurst_rs_intercept': 0.0,
            'hurst_stability': 0.0,
        }

    sizes = []
    rs_values = []
    size = min_window
    while size <= max_window:
        sizes.append(size)
        size = int(size * 1.5) + 1

    for size in sizes:
        rs_list = []
        for start in range(0, n - size + 1, max(size // 2, 1)):
            window = sent_lengths[start:start + size]
            mean_val = window.mean()
            deviations = window - mean_val
            cumsum = np.cumsum(deviations)
            r = cumsum.max() - cumsum.min()
            s = window.std(ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            rs_values.append(np.mean(rs_list))
        else:
            rs_values.append(0.0)

    # filter out zeros for log
    valid = [(s, rs) for s, rs in zip(sizes, rs_values) if rs > 0 and s > 0]
    if len(valid) < 2:
        return {
            'hurst_exponent': 0.5,
            'hurst_rs_intercept': 0.0,
            'hurst_stability': 0.0,
        }

    log_sizes = np.log(np.array([v[0] for v in valid]))
    log_rs = np.log(np.array([v[1] for v in valid]))

    # linear regression in log-log space: H = slope
    coeffs = np.polyfit(log_sizes, log_rs, 1)
    feats['hurst_exponent'] = np.clip(coeffs[0], 0.0, 1.5)
    feats['hurst_rs_intercept'] = coeffs[1]

    # stability: how well the log-log fit explains the data
    predicted = np.polyval(coeffs, log_sizes)
    ss_res = np.sum((log_rs - predicted) ** 2)
    ss_tot = np.sum((log_rs - log_rs.mean()) ** 2)
    feats['hurst_stability'] = 1.0 - ss_res / (ss_tot + 1e-10)

    return feats


# -- pairwise features (14) --

def pairwise_features(text_1, text_2):
    """Pairwise similarity/distance features between two texts (~14 features)."""
    feats = {}

    feats['ncd_gzip'] = _ncd(text_1, text_2, gzip.compress)

    feats['ncd_lzma'] = _ncd(text_1, text_2, lzma.compress)

    feats['ncd_bz2'] = _ncd(text_1, text_2, bz2.compress)

    words_1 = set(text_1.lower().split())
    words_2 = set(text_2.lower().split())

    union = words_1 | words_2
    intersection = words_1 & words_2
    feats['word_jaccard'] = len(intersection) / len(union) if union else 0.0

    fw_set = set(FUNCTION_WORDS)
    cw_1 = words_1 - fw_set
    cw_2 = words_2 - fw_set
    cw_union = cw_1 | cw_2
    cw_intersection = cw_1 & cw_2
    feats['content_word_overlap'] = len(cw_intersection) / len(cw_union) if cw_union else 0.0

    len_1 = len(text_1)
    len_2 = len(text_2)
    feats['length_ratio'] = min(len_1, len_2) / max(len_1, len_2) if max(len_1, len_2) > 0 else 1.0

    wc_1 = len(text_1.split())
    wc_2 = len(text_2.split())
    feats['word_count_ratio'] = min(wc_1, wc_2) / max(wc_1, wc_2) if max(wc_1, wc_2) > 0 else 1.0

    v_1 = len(words_1)
    v_2 = len(words_2)
    feats['vocab_size_ratio'] = min(v_1, v_2) / max(v_1, v_2) if max(v_1, v_2) > 0 else 1.0

    feats['jsd_word_freq'] = _jsd_word_freq(text_1, text_2)

    feats['jsd_char_bigram'] = _jsd_char_bigram(text_1, text_2)

    feats['burrows_delta'] = _burrows_delta(text_1, text_2)

    # Cosine Delta (Evert et al. 2017) -- cosine distance after z-score norm
    feats['cosine_delta'] = _cosine_delta(text_1, text_2)

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


def _cosine_delta(text_1, text_2, n_top=100):
    """Cosine Delta (Evert et al. 2017): cosine distance after z-score norm."""
    words_1 = text_1.lower().split()
    words_2 = text_2.lower().split()

    if len(words_1) < 200 or len(words_2) < 200:
        return 0.0

    freq_1 = Counter(words_1)
    freq_2 = Counter(words_2)
    n1 = len(words_1)
    n2 = len(words_2)

    combined = Counter(words_1 + words_2)
    top_words = [w for w, _ in combined.most_common(n_top)]

    all_words = words_1 + words_2
    combined_freq = Counter(all_words)
    n_total = len(all_words)

    # build z-scored vectors for each text
    z1 = []
    z2 = []
    for word in top_words:
        f1 = freq_1.get(word, 0) / n1
        f2 = freq_2.get(word, 0) / n2
        mean_f = combined_freq.get(word, 0) / n_total
        std_f = max(abs(f1 - mean_f), abs(f2 - mean_f))
        if std_f > 0:
            z1.append((f1 - mean_f) / std_f)
            z2.append((f2 - mean_f) / std_f)
        else:
            z1.append(0.0)
            z2.append(0.0)

    z1 = np.array(z1)
    z2 = np.array(z2)

    # cosine distance = 1 - cosine_similarity
    dot = np.dot(z1, z2)
    norm1 = np.linalg.norm(z1)
    norm2 = np.linalg.norm(z2)
    if norm1 > 0 and norm2 > 0:
        return 1.0 - dot / (norm1 * norm2)
    return 0.0


# -- combined feature extraction --

def extract_per_text_features(text):
    """All per-text features (groups 1-13). TF-IDF and spaCy are separate."""
    feats = {}
    feats.update(lexical_features(text))
    feats.update(character_features(text))
    feats.update(function_word_features(text))
    feats.update(structural_features(text))
    feats.update(writing_rhythm_features(text))
    feats.update(info_theoretic_features(text))
    # novel groups 10-13
    feats.update(spectral_features(text))
    feats.update(zipf_features(text))
    feats.update(benford_features(text))
    feats.update(hurst_features(text))
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
