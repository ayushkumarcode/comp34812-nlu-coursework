"""NLI Cat B iter2 — ESIM + KIM with LR=2e-4, patience=10, hidden=300.

Changes from v1:
- LR=2e-4 (was 4e-4) — slower, more stable learning
- Patience=10 (was 7) — more time to converge
- max_epochs=60 (was 40) — more capacity
- Clip grad norm=5.0 (was 10.0) — more stable
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.nli_cat_b_model import ESIM
from src.models.nli_cat_b_dataset import (
    NLIVocabulary, NLIESIMDataset, load_glove_embeddings
)
from src.data_utils import load_nli_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics


def train_epoch(model, dataloader, optimizer, criterion, device,
                use_wordnet=False):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for batch in dataloader:
        p_word = batch['premise_word_ids'].to(device)
        p_char = batch['premise_char_ids'].to(device)
        h_word = batch['hypothesis_word_ids'].to(device)
        h_char = batch['hypothesis_char_ids'].to(device)
        labels = batch['label'].to(device)
        wordnet = None
        if use_wordnet and 'wordnet_relations' in batch:
            wordnet = batch['wordnet_relations'].to(device)
        optimizer.zero_grad()
        logits = model(p_word, p_char, h_word, h_char, wordnet)
        loss = criterion(logits.squeeze(-1), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), f1_score(all_labels, all_preds, average='macro', zero_division=0)


def evaluate(model, dataloader, device, use_wordnet=False):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            p_word = batch['premise_word_ids'].to(device)
            p_char = batch['premise_char_ids'].to(device)
            h_word = batch['hypothesis_word_ids'].to(device)
            h_char = batch['hypothesis_char_ids'].to(device)
            labels = batch['label']
            wordnet = None
            if use_wordnet and 'wordnet_relations' in batch:
                wordnet = batch['wordnet_relations'].to(device)
            logits = model(p_word, p_char, h_word, h_char, wordnet)
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def main():
    print("=" * 60)
    print("  NLI Cat B v2 — ESIM + KIM (LR=2e-4, patience=10)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    HIDDEN_SIZE = 300
    BATCH_SIZE = 32
    MAX_EPOCHS = 60
    PATIENCE = 10
    LR = 2e-4
    PREMISE_MAX = 64
    HYPOTHESIS_MAX = 32
    USE_WORDNET = True
    GLOVE_PATH = str(Path.home() / 'scratch' / 'nlu-project' / 'glove.6B.300d.txt')

    print("\n[1/6] Loading data...")
    train_df = load_nli_data(split='train')
    dev_df = load_nli_data(split='dev')
    dev_labels = load_solution_labels(task='nli')

    print("\n[2/6] Building vocabulary...")
    vocab = NLIVocabulary(min_word_freq=2)
    all_texts = list(train_df['premise']) + list(train_df['hypothesis'])
    vocab.build_word_vocab(all_texts)

    import os
    pretrained_emb = None
    if os.path.exists(GLOVE_PATH):
        pretrained_emb = load_glove_embeddings(vocab, GLOVE_PATH, dim=300)
    emb_dim = pretrained_emb.shape[1] if pretrained_emb is not None else 300

    print("\n[3/6] Creating datasets...")
    train_dataset = NLIESIMDataset(
        train_df, vocab, PREMISE_MAX, HYPOTHESIS_MAX,
        compute_wordnet=USE_WORDNET)
    dev_dataset = NLIESIMDataset(
        dev_df, vocab, PREMISE_MAX, HYPOTHESIS_MAX,
        compute_wordnet=USE_WORDNET)
    dev_dataset.labels = np.array(dev_labels, dtype=np.float32)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)

    print("\n[4/6] Building model...")
    model = ESIM(
        vocab_size=vocab.vocab_size,
        embedding_dim=emb_dim,
        hidden_size=HIDDEN_SIZE,
        char_vocab_size=vocab.char_vocab_size,
        knowledge_dim=5,
        dropout=0.3,
        pretrained_embeddings=pretrained_emb,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                   patience=3, min_lr=1e-6)

    print("\n[5/6] Training...")
    best_f1 = 0.0
    patience_counter = 0
    save_dir = PROJECT_ROOT / 'models'
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        if epoch == 6 and pretrained_emb is not None:
            model.unfreeze_embeddings()
            print("  -> Word embeddings unfrozen")

        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_wordnet=USE_WORDNET)

        dev_preds, dev_probs, dev_true = evaluate(
            model, dev_loader, device, use_wordnet=USE_WORDNET)
        dev_f1 = f1_score(dev_true, dev_preds, average='macro', zero_division=0)
        scheduler.step(dev_f1)

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
              f"Train F1: {train_f1:.4f} | Dev F1: {dev_f1:.4f} | "
