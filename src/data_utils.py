"""
Data loading and preprocessing utilities for COMP34812 NLU Coursework.
Supports both AV (Authorship Verification) and NLI (Natural Language Inference) tracks.
"""

import html
import re
import unicodedata
from pathlib import Path

import pandas as pd

# Project root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "training_extracted" / "training_data"
SCORER_ROOT = PROJECT_ROOT / "baseline_extracted" / "nlu_bundle-feature-unified-local-scorer"
BASELINE_ROOT = SCORER_ROOT / "baseline"

# Data paths
AV_TRAIN_PATH = DATA_ROOT / "AV" / "train.csv"
AV_DEV_PATH = DATA_ROOT / "AV" / "dev.csv"
NLI_TRAIN_PATH = DATA_ROOT / "NLI" / "train.csv"
NLI_DEV_PATH = DATA_ROOT / "NLI" / "dev.csv"

# Baseline prediction paths
AV_BASELINE_PATH = BASELINE_ROOT / "25_DEV_AV.csv"
NLI_BASELINE_PATH = BASELINE_ROOT / "25_DEV_NLI.csv"

# URL regex pattern
URL_PATTERN = re.compile(
    r'https?://\S+|www\.\S+|ftp://\S+', re.IGNORECASE
)


def clean_text(text, lowercase=False):
    """Clean text while preserving stylistic signals.

    Args:
        text: Raw input text string.
        lowercase: If True, lowercase the text (used for NLI, not AV).

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Decode HTML entities
    text = html.unescape(text)

    # Normalize Unicode to NFC form
    text = unicodedata.normalize('NFC', text)

    # Replace URLs with <URL> token
    text = URL_PATTERN.sub('<URL>', text)

    # Normalize multiple spaces to single space
    text = re.sub(r' {2,}', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    if lowercase:
        text = text.lower()

    return text


def load_av_data(split='train'):
    """Load AV (Authorship Verification) data.

    Args:
        split: 'train' or 'dev'.

    Returns:
        DataFrame with columns: text_1, text_2, label (int).
        For dev split, label column may not be present.
    """
    path = AV_TRAIN_PATH if split == 'train' else AV_DEV_PATH
    df = pd.read_csv(path, quotechar='"', engine='python')

    # Clean texts (preserve case for AV stylometric features)
    df['text_1'] = df['text_1'].apply(lambda x: clean_text(x, lowercase=False))
    df['text_2'] = df['text_2'].apply(lambda x: clean_text(x, lowercase=False))

    # Convert labels to int if present
    if 'label' in df.columns:
        df['label'] = df['label'].astype(int)

    return df


def load_nli_data(split='train'):
    """Load NLI (Natural Language Inference) data.

    Args:
        split: 'train' or 'dev'.

    Returns:
        DataFrame with columns: premise, hypothesis, label (int).
        For dev split, label column may not be present.
    """
    path = NLI_TRAIN_PATH if split == 'train' else NLI_DEV_PATH
    df = pd.read_csv(path, quotechar='"', engine='python')

    # Clean texts (lowercase for NLI feature extraction)
    df['premise'] = df['premise'].apply(lambda x: clean_text(x, lowercase=False))
    df['hypothesis'] = df['hypothesis'].apply(lambda x: clean_text(x, lowercase=False))

    # Handle empty premise edge case (min word count = 0 in training)
    df['premise'] = df['premise'].apply(lambda x: '.' if not x or x.isspace() else x)

    # Labels should already be int for NLI
    if 'label' in df.columns:
        df['label'] = df['label'].astype(int)

    return df


def load_baseline_predictions(task='av'):
    """Load baseline predictions for McNemar's test.

    Args:
        task: 'av' or 'nli'.

    Returns:
        DataFrame with baseline model predictions.
    """
    path = AV_BASELINE_PATH if task == 'av' else NLI_BASELINE_PATH
    df = pd.read_csv(path)
    return df


def load_solution_labels(task='av'):
    """Load ground truth labels from the scorer's reference data.

    Args:
        task: 'av' or 'nli'.

    Returns:
        List of integer labels.
    """
    ref_dir = SCORER_ROOT / "local_scorer" / "reference_data"
    if task == 'av':
        sol_path = ref_dir / "NLU_SharedTask_AV_dev.solution"
    else:
        sol_path = ref_dir / "NLU_SharedTask_NLI_dev.solution"

    labels = []
    with open(sol_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    labels.append(int(float(line)))
                except ValueError:
                    continue
    return labels


def save_predictions(predictions, filepath):
    """Save predictions in the expected format (single column of integers).

    Args:
        predictions: List or array of integer predictions (0 or 1).
        filepath: Path to save the CSV file.
    """
    filepath = Path(filepath)
    with open(filepath, 'w') as f:
        for pred in predictions:
            f.write(f"{int(pred)}\n")


def get_data_stats(df, task='av'):
    """Print summary statistics for a dataset.

    Args:
        df: DataFrame loaded from load_av_data or load_nli_data.
        task: 'av' or 'nli'.
    """
    print(f"Dataset shape: {df.shape}")

    if 'label' in df.columns:
        print(f"Label distribution:\n{df['label'].value_counts()}")
        print(f"Label proportions:\n{df['label'].value_counts(normalize=True)}")

    if task == 'av':
        for col in ['text_1', 'text_2']:
            word_counts = df[col].apply(lambda x: len(x.split()))
            print(f"\n{col} word counts:")
            print(f"  min={word_counts.min()}, max={word_counts.max()}, "
                  f"mean={word_counts.mean():.1f}, median={word_counts.median():.1f}")
    else:
        for col in ['premise', 'hypothesis']:
            word_counts = df[col].apply(lambda x: len(x.split()))
            print(f"\n{col} word counts:")
            print(f"  min={word_counts.min()}, max={word_counts.max()}, "
                  f"mean={word_counts.mean():.1f}, median={word_counts.median():.1f}")
