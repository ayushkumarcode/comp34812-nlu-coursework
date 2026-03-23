"""
NLI Category B — Dataset for ESIM + KIM.
Handles word/char tokenization, vocabulary building, and WordNet relations.
"""

import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import Counter


PAD_IDX = 0
UNK_IDX = 1


class NLIVocabulary:
    """Word and character vocabulary for NLI."""

    def __init__(self, min_word_freq=2):
        self.min_word_freq = min_word_freq
        self.word2idx = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
        self.idx2word = {PAD_IDX: '<PAD>', UNK_IDX: '<UNK>'}
        self.char2idx = {'<PAD>': 0, '<UNK>': 1}

        # Build char vocab from printable ASCII
        for i, c in enumerate(range(32, 127)):
            self.char2idx[chr(c)] = i + 2
        self.char_vocab_size = len(self.char2idx)

    def build_word_vocab(self, texts):
        """Build word vocabulary from a list of texts.

        Args:
            texts: List of text strings.
        """
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)

        idx = len(self.word2idx)
        for word, count in word_counts.most_common():
            if count >= self.min_word_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        print(f"Word vocabulary size: {len(self.word2idx)}")

    def encode_words(self, text, max_len):
        """Encode text to word indices."""
        words = self._tokenize(text)[:max_len]
        indices = [self.word2idx.get(w, UNK_IDX) for w in words]
        # Pad
        indices.extend([PAD_IDX] * (max_len - len(indices)))
        return np.array(indices, dtype=np.int64)

    def encode_chars(self, text, max_word_len, max_char_len=16):
        """Encode text to character indices per word."""
        words = self._tokenize(text)[:max_word_len]
        result = np.zeros((max_word_len, max_char_len), dtype=np.int64)

        for i, word in enumerate(words):
            for j, c in enumerate(word[:max_char_len]):
                result[i, j] = self.char2idx.get(c, 1)

        return result

    def _tokenize(self, text):
        """Simple whitespace + punctuation tokenization."""
        text = text.lower()
        # Replace digits with 0 for better generalization
        text = re.sub(r'\d', '0', text)
        # Split on whitespace and separate punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

    @property
    def vocab_size(self):
        return len(self.word2idx)


class NLIESIMDataset(Dataset):
    """PyTorch Dataset for NLI ESIM model."""

    def __init__(self, df, vocab, premise_max_len=64, hypothesis_max_len=32,
                 char_max_len=16, compute_wordnet=False):
        """
        Args:
            df: DataFrame with premise, hypothesis, label columns.
            vocab: NLIVocabulary instance.
            premise_max_len: Max premise token length.
            hypothesis_max_len: Max hypothesis token length.
            char_max_len: Max characters per word.
            compute_wordnet: Whether to pre-compute WordNet relations.
        """
        self.premises = list(df['premise'])
        self.hypotheses = list(df['hypothesis'])
        self.labels = df['label'].values.astype(np.float32)
        self.vocab = vocab
        self.p_max = premise_max_len
        self.h_max = hypothesis_max_len
        self.char_max = char_max_len

        # Pre-encode
        print("Pre-encoding premises...")
        self.p_word_ids = [vocab.encode_words(t, self.p_max) for t in self.premises]
        self.p_char_ids = [vocab.encode_chars(t, self.p_max, char_max_len) for t in self.premises]

        print("Pre-encoding hypotheses...")
        self.h_word_ids = [vocab.encode_words(t, self.h_max) for t in self.hypotheses]
        self.h_char_ids = [vocab.encode_chars(t, self.h_max, char_max_len) for t in self.hypotheses]

        # WordNet relations (optional)
        self.wordnet_relations = None
        if compute_wordnet:
            print("Pre-computing WordNet relations...")
            self.wordnet_relations = self._compute_wordnet_relations()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'premise_word_ids': torch.tensor(self.p_word_ids[idx]),
            'premise_char_ids': torch.tensor(self.p_char_ids[idx]),
            'hypothesis_word_ids': torch.tensor(self.h_word_ids[idx]),
            'hypothesis_char_ids': torch.tensor(self.h_char_ids[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.float),
        }

        if self.wordnet_relations is not None:
            item['wordnet_relations'] = torch.tensor(
                self.wordnet_relations[idx], dtype=torch.float
            )

        return item

    def _compute_wordnet_relations(self):
        """Pre-compute WordNet relations for all pairs."""
        try:
            from nltk.corpus import wordnet as wn
        except ImportError:
            print("Warning: NLTK WordNet not available, skipping relations")
            return None

        relations = []
        for i in range(len(self.premises)):
            p_words = self.vocab._tokenize(self.premises[i])[:self.p_max]
            h_words = self.vocab._tokenize(self.hypotheses[i])[:self.h_max]

            rel_matrix = np.zeros((self.p_max, self.h_max, 5), dtype=np.float32)

            for pi, pw in enumerate(p_words):
                p_synsets = wn.synsets(pw)
                if not p_synsets:
                    continue
                for hi, hw in enumerate(h_words):
                    h_synsets = wn.synsets(hw)
                    if not h_synsets:
                        continue
                    rel_matrix[pi, hi] = self._get_relations(
                        pw, hw, p_synsets, h_synsets
                    )

            relations.append(rel_matrix)

            if (i + 1) % 5000 == 0:
                print(f"  WordNet relations: {i+1}/{len(self.premises)}")

        return relations

    def _get_relations(self, word_p, word_h, synsets_p, synsets_h):
        """Get binary relation vector [syn, ant, hyper, hypo, cohypo]."""
        rel = np.zeros(5, dtype=np.float32)

        p_lemmas = set()
        h_lemmas = set()
        for s in synsets_p:
            p_lemmas.update(l.name() for l in s.lemmas())
        for s in synsets_h:
            h_lemmas.update(l.name() for l in s.lemmas())

        # Synonym
        if p_lemmas & h_lemmas:
            rel[0] = 1.0

        # Antonym
        for s in synsets_p:
            for l in s.lemmas():
                for ant in l.antonyms():
                    if ant.name() in h_lemmas:
                        rel[1] = 1.0

        # Hypernym (p is hypernym of h)
        for sp in synsets_p:
            for sh in synsets_h:
                if sp in sh.lowest_common_hypernyms(sp):
                    rel[2] = 1.0

        # Hyponym (p is hyponym of h)
        for sp in synsets_p:
            for sh in synsets_h:
                if sh in sp.lowest_common_hypernyms(sh):
                    rel[3] = 1.0

        # Co-hyponym
        for sp in synsets_p:
            for sh in synsets_h:
                p_hypers = set(sp.hypernyms())
                h_hypers = set(sh.hypernyms())
                if p_hypers & h_hypers:
                    rel[4] = 1.0

        return rel


def load_glove_embeddings(vocab, glove_path, dim=300):
    """Load GloVe embeddings for vocabulary.

    Returns:
        torch.FloatTensor of shape (vocab_size, dim).
    """
    embeddings = torch.zeros(vocab.vocab_size, dim)
    nn.init.uniform_(embeddings, -0.05, 0.05)
    embeddings[PAD_IDX] = 0

    loaded = 0
    print(f"Loading GloVe from {glove_path}...")
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            word = parts[0]
            if word in vocab.word2idx:
                vec = torch.tensor([float(x) for x in parts[1:]])
                if len(vec) == dim:
                    embeddings[vocab.word2idx[word]] = vec
                    loaded += 1

    print(f"Loaded {loaded}/{vocab.vocab_size} word embeddings from GloVe.")
    return embeddings
