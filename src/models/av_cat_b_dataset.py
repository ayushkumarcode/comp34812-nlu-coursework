"""
Char-level dataset for Cat B. Handles encoding, augmentation, and topic labels.
"""

import re
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans


CHAR_VOCAB = {chr(c): i+1 for i, c in enumerate(range(ord('a'), ord('z')+1))}
for i, c in enumerate(range(ord('A'), ord('Z')+1)):
    CHAR_VOCAB[chr(c)] = 27 + i
for i, c in enumerate(range(ord('0'), ord('9')+1)):
    CHAR_VOCAB[chr(c)] = 53 + i
SPECIAL_CHARS = list('.,;:!?\'"()-/\\@#$%&*_+=<>[]{}|~ \t\n')
for i, c in enumerate(SPECIAL_CHARS):
    CHAR_VOCAB[c] = 63 + i
UNK_IDX = len(CHAR_VOCAB) + 1
PAD_IDX = 0
VOCAB_SIZE = UNK_IDX + 1  # ~97


def char_encode(text, max_len=1500):
    """Turn text into a fixed-length array of char indices."""
    indices = []
    for c in text[:max_len]:
        indices.append(CHAR_VOCAB.get(c, UNK_IDX))

    if len(indices) < max_len:
        indices.extend([PAD_IDX] * (max_len - len(indices)))

    return np.array(indices, dtype=np.int64)


def augment_text(char_ids, perturb_prob=0.05, truncate_range=(0.8, 1.0)):
    """Augment by randomly truncating and perturbing characters."""
    ids = char_ids.copy()

    non_pad = np.sum(ids > 0)
    if non_pad > 10:
        keep_ratio = np.random.uniform(*truncate_range)
        keep_len = int(non_pad * keep_ratio)
        ids[keep_len:] = PAD_IDX

    mask = np.random.random(len(ids)) < perturb_prob
    mask &= (ids > 0)

    for i in np.where(mask)[0]:
        c = ids[i]
        if 1 <= c <= 26:  # lowercase letter
            ids[i] = np.random.randint(1, 27)
        elif 27 <= c <= 52:  # uppercase letter
            ids[i] = np.random.randint(27, 53)
        elif 53 <= c <= 62:  # digit
            ids[i] = np.random.randint(53, 63)

    return ids


class AVCharDataset(Dataset):
    """PyTorch dataset for char-level AV data with optional augmentation."""

    def __init__(self, df, max_len=1500, augment=False,
                 topic_labels=None):
        """
        Args:
            df: DataFrame with text_1, text_2, label columns.
            max_len: Max character sequence length.
            augment: Whether to apply augmentation during training.
            topic_labels: Optional array of topic pseudo-labels.
        """
        self.texts_1 = list(df['text_1'])
        self.texts_2 = list(df['text_2'])
        self.labels = df['label'].values.astype(np.float32)
        self.max_len = max_len
        self.augment = augment
        self.topic_labels = topic_labels

        self.encoded_1 = [char_encode(t, max_len) for t in self.texts_1]
        self.encoded_2 = [char_encode(t, max_len) for t in self.texts_2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids_1 = self.encoded_1[idx]
        ids_2 = self.encoded_2[idx]

        if self.augment:
            ids_1 = augment_text(ids_1)
            ids_2 = augment_text(ids_2)

        item = {
            'char_ids_1': torch.tensor(ids_1, dtype=torch.long),
            'char_ids_2': torch.tensor(ids_2, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.float),
        }

        if self.topic_labels is not None:
            item['topic'] = torch.tensor(self.topic_labels[idx], dtype=torch.long)

        return item


def generate_topic_labels(texts, n_clusters=10):
    """Generate topic pseudo-labels. Tries heuristic first, falls back to K-Means."""
    labels = []
    for text in texts:
        if _is_email(text):
            labels.append(0)
        elif _is_blog(text):
            labels.append(1)
        elif _is_review(text):
            labels.append(2)
        else:
            labels.append(3)

    labels = np.array(labels)
    unique_counts = Counter(labels)

    if len(unique_counts) >= 3 and min(unique_counts.values()) > len(texts) * 0.05:
        print(f"Using heuristic topic labels: {dict(unique_counts)}")
        return labels

    print("Falling back to TF-IDF + K-Means clustering for topic labels...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    features = tfidf.fit_transform(texts)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    cluster_labels = kmeans.fit_predict(features)
    print(f"Cluster distribution: {dict(Counter(cluster_labels))}")
    return cluster_labels


def _is_email(text):
    """Looks for email headers (From/To/Subject lines)."""
    return bool(re.search(r'^(From|To|Subject|Date|Sent):', text, re.MULTILINE))


def _is_blog(text):
    """Blog posts often have urlLink markers."""
    return 'urlLink' in text or 'urllink' in text.lower()


def _is_review(text):
    """Checks for review-ish patterns like ratings and stars."""
    review_patterns = [
        r'\b\d+/\d+\b',  # ratings like 8/10
        r'\bstars?\b',
        r'\brating\b',
        r'\brecommend\b',
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in review_patterns)
