"""
spaCy-based features: POS tag distributions (group 5) and
syntactic complexity measures (group 7).
"""

import numpy as np
from collections import Counter


UPOS_TAGS = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
    'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'
]


def get_spacy_model():
    """Load spaCy model, falls back to sm if md isn't installed."""
    import spacy
    try:
        nlp = spacy.load('en_core_web_md', disable=['ner'])
    except OSError:
        nlp = spacy.load('en_core_web_sm', disable=['ner'])
    return nlp


def pos_features(doc):
    """POS tag frequencies (17 UPOS) + POS bigram frequencies (28) = 45 features."""
    feats = {}
    tokens = [t for t in doc if not t.is_space]
    n_tokens = max(len(tokens), 1)

    # POS tag frequencies
    pos_counts = Counter(t.pos_ for t in tokens)
    for tag in UPOS_TAGS:
        feats[f'pos_{tag}'] = pos_counts.get(tag, 0) / n_tokens

    # POS bigram frequencies (top 28)
    if len(tokens) > 1:
        pos_seq = [t.pos_ for t in tokens]
        bigrams = list(zip(pos_seq[:-1], pos_seq[1:]))
        bigram_counts = Counter(bigrams)
        total_bigrams = len(bigrams)
        # Use predefined common bigrams
        for bg in _TOP_POS_BIGRAMS:
            feats[f'pos_bg_{bg[0]}_{bg[1]}'] = bigram_counts.get(bg, 0) / total_bigrams
    else:
        for bg in _TOP_POS_BIGRAMS:
            feats[f'pos_bg_{bg[0]}_{bg[1]}'] = 0.0

    return feats


# Top 28 POS bigrams (common in English text)
_TOP_POS_BIGRAMS = [
    ('DET', 'NOUN'), ('ADJ', 'NOUN'), ('NOUN', 'VERB'), ('NOUN', 'ADP'),
    ('ADP', 'DET'), ('VERB', 'DET'), ('PRON', 'VERB'), ('ADP', 'NOUN'),
    ('VERB', 'ADP'), ('ADV', 'VERB'), ('NOUN', 'NOUN'), ('DET', 'ADJ'),
    ('VERB', 'VERB'), ('AUX', 'VERB'), ('VERB', 'PRON'), ('NOUN', 'PUNCT'),
    ('PUNCT', 'DET'), ('PUNCT', 'PRON'), ('VERB', 'ADV'), ('ADV', 'ADJ'),
    ('PRON', 'AUX'), ('ADP', 'PRON'), ('PUNCT', 'CCONJ'), ('CCONJ', 'PRON'),
    ('VERB', 'NOUN'), ('PUNCT', 'ADV'), ('NOUN', 'CCONJ'), ('ADJ', 'PUNCT'),
]


def syntactic_complexity_features(doc):
    """Extract syntactic complexity features from a spaCy Doc.

    Group 7 — NOVEL for AV. 10 features.
    """
    feats = {}
    sents = list(doc.sents)
    n_sents = max(len(sents), 1)

    if not sents:
        return {k: 0.0 for k in [
            'avg_dep_depth', 'max_dep_depth', 'avg_branching_factor',
            'subordination_index', 'avg_dep_arc_length', 'passive_ratio',
            'relcl_ratio', 'avg_conjuncts', 'content_clause_ratio',
            'fronted_adverb_ratio',
        ]}

    # Dependency parse depths
    depths = []
    max_depths = []
    branching_factors = []

    for sent in sents:
        root = sent.root
        sent_depth = _tree_depth(root)
        max_depths.append(sent_depth)
        depths.append(sent_depth)

        # Branching factor
        non_leaf_children = []
        for token in sent:
            children = list(token.children)
            if children:
                non_leaf_children.append(len(children))
        if non_leaf_children:
            branching_factors.append(np.mean(non_leaf_children))

    feats['avg_dep_depth'] = np.mean(depths) if depths else 0.0
    feats['max_dep_depth'] = max(max_depths) if max_depths else 0.0
    feats['avg_branching_factor'] = np.mean(branching_factors) if branching_factors else 0.0

    # Subordination index (SCONJ-headed clauses per sentence)
    sconj_count = sum(1 for t in doc if t.dep_ == 'mark' and t.pos_ == 'SCONJ')
    feats['subordination_index'] = sconj_count / n_sents

    # Average dependency arc length
    arc_lengths = [abs(t.i - t.head.i) for t in doc if t.dep_ != 'ROOT' and not t.is_space]
    feats['avg_dep_arc_length'] = np.mean(arc_lengths) if arc_lengths else 0.0

    # Passive construction ratio
    passive_count = sum(1 for t in doc if t.dep_ == 'nsubjpass' or t.dep_ == 'auxpass')
    all_clauses = max(sum(1 for t in doc if t.dep_ == 'nsubj' or t.dep_ == 'nsubjpass'), 1)
    feats['passive_ratio'] = passive_count / all_clauses

    # Relative clause count per sentence
    relcl_count = sum(1 for t in doc if t.dep_ == 'relcl')
    feats['relcl_ratio'] = relcl_count / n_sents

    # Average conjuncts per coordination
    coord_heads = [t for t in doc if any(c.dep_ == 'conj' for c in t.children)]
    if coord_heads:
        conj_counts = [sum(1 for c in t.children if c.dep_ == 'conj') + 1 for t in coord_heads]
        feats['avg_conjuncts'] = np.mean(conj_counts)
    else:
        feats['avg_conjuncts'] = 0.0

    # Content clause ratio (ccomp + xcomp per sentence)
    cc_count = sum(1 for t in doc if t.dep_ in ('ccomp', 'xcomp'))
    feats['content_clause_ratio'] = cc_count / n_sents

    # Fronted adverbials (advmod before root verb)
    fronted = 0
    for sent in sents:
        root = sent.root
        for child in root.children:
            if child.dep_ == 'advmod' and child.i < root.i:
                fronted += 1
    feats['fronted_adverb_ratio'] = fronted / n_sents

    return feats


def _tree_depth(token):
    """Recursively compute depth of dependency subtree."""
    children = list(token.children)
    if not children:
        return 0
    return 1 + max(_tree_depth(c) for c in children)


def extract_spacy_features(text, nlp):
    """Extract all spaCy-dependent features for a single text.

    Args:
        text: Raw text string.
        nlp: spaCy Language model.

    Returns:
        Dict of feature name -> float value. ~55 features.
    """
    doc = nlp(text)
    feats = {}
    feats.update(pos_features(doc))
    feats.update(syntactic_complexity_features(doc))
    return feats


def batch_extract_spacy_features(texts, nlp, batch_size=256):
    """Extract spaCy features for a batch of texts.

    Args:
        texts: List of text strings.
        nlp: spaCy Language model.
        batch_size: Batch size for spaCy.pipe().

    Returns:
        List of dicts, one per text.
    """
    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        feats = {}
        feats.update(pos_features(doc))
        feats.update(syntactic_complexity_features(doc))
        results.append(feats)
    return results
