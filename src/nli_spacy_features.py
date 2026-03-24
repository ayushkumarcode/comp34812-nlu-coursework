"""
NLI Feature Engineering — spaCy-dependent features.
Tier 4: Syntactic features (20)
Tier 5: Alignment-based features — Sultan et al. 2014 inspired (12)
Tier 6: Natural logic features — MacCartney & Manning inspired (8)
"""

import numpy as np
from collections import Counter


# Universal POS tags
UPOS_TAGS = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
    'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'
]


def get_spacy_model():
    """Load and return spaCy model."""
    import spacy
    try:
        nlp = spacy.load('en_core_web_md', disable=['textcat'])
    except OSError:
        nlp = spacy.load('en_core_web_sm', disable=['textcat'])
    return nlp


# ============================================================
# TIER 4: SYNTACTIC FEATURES (20 features)
# ============================================================

def syntactic_features(p_doc, h_doc):
    """Extract syntactic features from spaCy docs.

    Returns:
        Dict with ~20 features: 17 POS diff + 3 structural.
    """
    feats = {}

    p_tokens = [t for t in p_doc if not t.is_space]
    h_tokens = [t for t in h_doc if not t.is_space]
    n_p = max(len(p_tokens), 1)
    n_h = max(len(h_tokens), 1)

    # POS tag distribution difference (17 features)
    p_pos = Counter(t.pos_ for t in p_tokens)
    h_pos = Counter(t.pos_ for t in h_tokens)
    for tag in UPOS_TAGS:
        p_freq = p_pos.get(tag, 0) / n_p
        h_freq = h_pos.get(tag, 0) / n_h
        feats[f'pos_diff_{tag}'] = abs(p_freq - h_freq)

    # Dependency tree depth difference
    p_depth = _tree_max_depth(p_doc)
    h_depth = _tree_max_depth(h_doc)
    feats['dep_depth_diff'] = abs(p_depth - h_depth)

    # Root verb match
    p_root = _get_root_lemma(p_doc)
    h_root = _get_root_lemma(h_doc)
    feats['root_verb_match'] = 1.0 if p_root and h_root and p_root == h_root else 0.0

    # SVO alignment score
    feats['svo_alignment'] = _svo_alignment(p_doc, h_doc)

    return feats


def _tree_max_depth(doc):
    """Get max dependency tree depth across sentences."""
    max_d = 0
    for sent in doc.sents:
        d = _subtree_depth(sent.root)
        max_d = max(max_d, d)
    return max_d


def _subtree_depth(token):
    """Recursively compute depth of dependency subtree."""
    children = list(token.children)
    if not children:
        return 0
    return 1 + max(_subtree_depth(c) for c in children)


def _get_root_lemma(doc):
    """Get lemma of the root verb."""
    for sent in doc.sents:
        root = sent.root
        if root.pos_ == 'VERB':
            return root.lemma_.lower()
    return None


def _svo_alignment(p_doc, h_doc):
    """Compute SVO alignment score between premise and hypothesis."""
    p_svo = _extract_svo(p_doc)
    h_svo = _extract_svo(h_doc)

    if not p_svo or not h_svo:
        return 0.0

    scores = []
    for component in ['S', 'V', 'O']:
        p_items = {item.lower() for svo in p_svo for item in svo.get(component, [])}
        h_items = {item.lower() for svo in h_svo for item in svo.get(component, [])}
        if p_items or h_items:
            inter = p_items & h_items
            union = p_items | h_items
            scores.append(len(inter) / max(len(union), 1))
    return np.mean(scores) if scores else 0.0


def _extract_svo(doc):
    """Extract subject-verb-object triples from a spaCy doc."""
    triples = []
    for sent in doc.sents:
        root = sent.root
        subj = [c.lemma_ for c in root.children if c.dep_ in ('nsubj', 'nsubjpass')]
        obj = [c.lemma_ for c in root.children if c.dep_ in ('dobj', 'pobj', 'attr')]
        triples.append({'S': subj, 'V': [root.lemma_], 'O': obj})
    return triples


# ============================================================
# TIER 5: ALIGNMENT-BASED FEATURES (12 features)
# ============================================================

def alignment_features(p_doc, h_doc):
    """Extract monolingual word alignment features.

    Sultan et al. 2014 inspired: exact match, lemma match,
    synonym match (via simple heuristic), contextual match.

    Returns:
        Dict with 12 features.
    """
    feats = {}

    p_tokens = [t for t in p_doc if not t.is_space and not t.is_punct]
    h_tokens = [t for t in h_doc if not t.is_space and not t.is_punct]

    if not p_tokens or not h_tokens:
        return _empty_alignment()

    # Alignment tracking
    p_aligned = [False] * len(p_tokens)
    h_aligned = [False] * len(h_tokens)
    alignments = []  # (p_idx, h_idx, type, quality)

    # Step 1: Exact match alignment
    for i, pt in enumerate(p_tokens):
        if p_aligned[i]:
            continue
        for j, ht in enumerate(h_tokens):
            if h_aligned[j]:
                continue
            if pt.text.lower() == ht.text.lower():
                p_aligned[i] = True
                h_aligned[j] = True
                alignments.append((i, j, 'exact', 1.0))
                break

    # Step 2: Lemma match alignment
    for i, pt in enumerate(p_tokens):
        if p_aligned[i]:
            continue
        for j, ht in enumerate(h_tokens):
            if h_aligned[j]:
                continue
            if pt.lemma_.lower() == ht.lemma_.lower():
                p_aligned[i] = True
                h_aligned[j] = True
                alignments.append((i, j, 'lemma', 0.9))
                break

    # Step 3: Contextual match (same POS + neighboring aligned words)
    for i, pt in enumerate(p_tokens):
        if p_aligned[i]:
            continue
        for j, ht in enumerate(h_tokens):
            if h_aligned[j]:
                continue
            if pt.pos_ == ht.pos_:
                # Check if neighboring words are aligned
                has_neighbor = False
                for di in [-1, 1]:
                    ni, nj = i + di, j + di
                    if 0 <= ni < len(p_tokens) and 0 <= nj < len(h_tokens):
                        if p_aligned[ni] and h_aligned[nj]:
                            has_neighbor = True
                            break
                if has_neighbor:
                    p_aligned[i] = True
                    h_aligned[j] = True
                    alignments.append((i, j, 'contextual', 0.6))
                    break

    # Compute features from alignment
    n_p = len(p_tokens)
    n_h = len(h_tokens)
    n_aligned = len(alignments)

    feats['alignment_coverage_p'] = sum(p_aligned) / max(n_p, 1)
    feats['alignment_coverage_h'] = sum(h_aligned) / max(n_h, 1)

    # Content word alignment coverage
    p_content = [i for i, t in enumerate(p_tokens) if t.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV')]
    h_content = [i for i, t in enumerate(h_tokens) if t.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV')]
    p_content_aligned = sum(1 for i in p_content if p_aligned[i])
    h_content_aligned = sum(1 for i in h_content if h_aligned[i])
    feats['content_alignment_coverage_p'] = p_content_aligned / max(len(p_content), 1)
    feats['content_alignment_coverage_h'] = h_content_aligned / max(len(h_content), 1)

    feats['aligned_pair_count'] = n_aligned

    # Alignment type proportions
    type_counts = Counter(a[2] for a in alignments)
    feats['exact_aligned_ratio'] = type_counts.get('exact', 0) / max(n_aligned, 1)
    feats['lemma_aligned_ratio'] = type_counts.get('lemma', 0) / max(n_aligned, 1)
    feats['synonym_aligned_ratio'] = type_counts.get('synonym', 0) / max(n_aligned, 1)
    feats['contextual_aligned_ratio'] = type_counts.get('contextual', 0) / max(n_aligned, 1)

    # Average alignment confidence
    if alignments:
        feats['avg_alignment_confidence'] = np.mean([a[3] for a in alignments])
    else:
        feats['avg_alignment_confidence'] = 0.0

    # Alignment symmetry
    feats['alignment_symmetry'] = abs(feats['alignment_coverage_p'] - feats['alignment_coverage_h'])

    # Unaligned hypothesis content words count
    feats['unaligned_h_content'] = sum(1 for i in h_content if not h_aligned[i])

    return feats


def _empty_alignment():
    """Return zeroed alignment features."""
    return {
        'alignment_coverage_p': 0.0,
        'alignment_coverage_h': 0.0,
        'content_alignment_coverage_p': 0.0,
        'content_alignment_coverage_h': 0.0,
        'aligned_pair_count': 0.0,
        'exact_aligned_ratio': 0.0,
        'lemma_aligned_ratio': 0.0,
        'contextual_aligned_ratio': 0.0,
        'avg_alignment_confidence': 0.0,
        'alignment_symmetry': 0.0,
        'unaligned_h_content': 0.0,
        'synonym_aligned_ratio': 0.0,
    }


# ============================================================
# TIER 6: NATURAL LOGIC FEATURES (8 features)
# ============================================================

def natural_logic_features(p_doc, h_doc):
    """Extract natural logic relation features.

    MacCartney & Manning 2007/2009 inspired.
    Classifies aligned word pairs into lexical relation types.

    Returns:
        Dict with 8 features.
    """
    feats = {}

    p_tokens = [t for t in p_doc if not t.is_space and not t.is_punct]
    h_tokens = [t for t in h_doc if not t.is_space and not t.is_punct]

    if not p_tokens or not h_tokens:
        return {
            'natlog_equiv_ratio': 0.0, 'natlog_fwd_ratio': 0.0,
            'natlog_rev_ratio': 0.0, 'natlog_alt_ratio': 0.0,
            'natlog_indep_ratio': 0.0, 'natlog_cover_ratio': 0.0,
            'entailment_score': 0.0, 'contradiction_score': 0.0,
        }

    # Classify relations for matched pairs
    relations = []
    for pt in p_tokens:
        for ht in h_tokens:
            if pt.text.lower() == ht.text.lower() or pt.lemma_.lower() == ht.lemma_.lower():
                rel = _classify_relation(pt, ht)
                relations.append(rel)

    n_rels = max(len(relations), 1)
    rel_counts = Counter(relations)

    feats['natlog_equiv_ratio'] = rel_counts.get('equiv', 0) / n_rels
    feats['natlog_fwd_ratio'] = rel_counts.get('forward', 0) / n_rels
    feats['natlog_rev_ratio'] = rel_counts.get('reverse', 0) / n_rels
    feats['natlog_alt_ratio'] = rel_counts.get('alternation', 0) / n_rels
    feats['natlog_indep_ratio'] = rel_counts.get('independence', 0) / n_rels
    feats['natlog_cover_ratio'] = rel_counts.get('cover', 0) / n_rels

    # Entailment and contradiction scores
    feats['entailment_score'] = (
        rel_counts.get('equiv', 0) * 1.0 +
        rel_counts.get('forward', 0) * 0.8
    ) / n_rels
    feats['contradiction_score'] = (
        rel_counts.get('alternation', 0) * 1.0 +
        rel_counts.get('independence', 0) * 0.3
    ) / n_rels

    return feats


def _classify_relation(p_token, h_token):
    """Classify lexical relation between aligned word pair using WordNet."""
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return 'independence'

    p_word = p_token.lemma_.lower()
    h_word = h_token.lemma_.lower()

    if p_word == h_word:
        return 'equiv'

    p_synsets = wn.synsets(p_word)
    h_synsets = wn.synsets(h_word)

    if not p_synsets or not h_synsets:
        return 'independence'

    # Check synonymy (shared lemma names across synsets)
    p_lemmas = set()
    h_lemmas = set()
    for s in p_synsets:
        p_lemmas.update(l.name() for l in s.lemmas())
    for s in h_synsets:
        h_lemmas.update(l.name() for l in s.lemmas())
    if p_lemmas & h_lemmas:
        return 'equiv'

    # Check antonymy
    for s in p_synsets:
        for l in s.lemmas():
            for ant in l.antonyms():
                if ant.name() in h_lemmas:
                    return 'alternation'

    return 'independence'


# ============================================================
# NER FEATURES
# ============================================================

def ner_features(p_doc, h_doc):
    """Extract named entity overlap features.

    Returns:
        Dict with 2 features.
    """
    feats = {}

    p_ents = {ent.text.lower() for ent in p_doc.ents}
    h_ents = {ent.text.lower() for ent in h_doc.ents}

    feats['entity_mismatch'] = 1.0 if h_ents and not h_ents.issubset(p_ents) else 0.0
    feats['entity_overlap_ratio'] = (
        len(p_ents & h_ents) / max(len(h_ents), 1) if h_ents else 0.0
    )

    return feats


# ============================================================
# COMBINED SPACY FEATURE EXTRACTION
# ============================================================

def extract_spacy_features(premise, hypothesis, nlp):
    """Extract all spaCy-dependent features for a single pair.

    Args:
        premise: Premise text string.
        hypothesis: Hypothesis text string.
        nlp: spaCy Language model.

    Returns:
        Dict of feature name -> float value.
    """
    p_doc = nlp(premise)
    h_doc = nlp(hypothesis)

    feats = {}
    feats.update(syntactic_features(p_doc, h_doc))
    feats.update(alignment_features(p_doc, h_doc))
    feats.update(natural_logic_features(p_doc, h_doc))
    feats.update(ner_features(p_doc, h_doc))

    return feats


def batch_extract_spacy_features(premises, hypotheses, nlp, batch_size=256):
    """Extract spaCy features for batches of premise-hypothesis pairs.

    Args:
        premises: List of premise strings.
        hypotheses: List of hypothesis strings.
        nlp: spaCy Language model.
        batch_size: Batch size for spaCy.pipe().

    Returns:
        List of feature dicts, one per pair.
    """
    # Process all texts through spaCy pipeline
    all_texts = list(premises) + list(hypotheses)
    all_docs = list(nlp.pipe(all_texts, batch_size=batch_size))
    n = len(premises)
    p_docs = all_docs[:n]
    h_docs = all_docs[n:]

    results = []
    for p_doc, h_doc in zip(p_docs, h_docs):
        feats = {}
        feats.update(syntactic_features(p_doc, h_doc))
        feats.update(alignment_features(p_doc, h_doc))
        feats.update(natural_logic_features(p_doc, h_doc))
        feats.update(ner_features(p_doc, h_doc))
        results.append(feats)

    return results
