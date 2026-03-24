"""AV Cat C — Cross-Encoder DeBERTa v2 with ScalarMix + GRL topic debiasing.

Creativity elements:
  1. ScalarMix layer weighting (Peters et al. 2018, ELMo-style)
  2. Gradient Reversal Layer for topic debiasing (Ganin et al. 2016)
  3. TF-IDF + KMeans topic pseudo-labels
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_av_data, load_solution_labels, save_predictions
from src.scorer import compute_all_metrics, print_metrics
from src.models.cat_c_deberta import ScalarMix, GRL
from src.models.av_cat_b_dataset import generate_topic_labels


class AVCrossEncoderDatasetV2(Dataset):
    """AV dataset for cross-encoder with topic pseudo-labels."""
    def __init__(self, df, tokenizer, max_len=384, topic_labels=None):
        self.texts_1 = list(df['text_1'])
        self.texts_2 = list(df['text_2'])
        self.labels = df['label'].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.topic_labels = topic_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts_1[idx], self.texts_2[idx],
            truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt'
        )
        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float),
        }
        if self.topic_labels is not None:
            item['topic'] = torch.tensor(self.topic_labels[idx], dtype=torch.long)
        return item


class AVCrossEncoderV2(nn.Module):
    """Cross-encoder with ScalarMix layer weighting + GRL topic head."""

    def __init__(self, encoder, num_topics=10, grl_lambda=0.05):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        num_layers = encoder.config.num_hidden_layers
        self.scalar_mix = ScalarMix(num_layers, style_bias=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(hidden_size, 256),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1),
        )
        self.grl = GRL(grl_lambda)
        self.topic_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(64, num_topics),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[1:]  # skip embedding layer
        cls_per_layer = [h[:, 0, :] for h in hidden_states]
        normed_w = F.softmax(self.scalar_mix.weights, dim=0)
        cls_repr = sum(w * c for w, c in zip(normed_w, cls_per_layer))
        logits = self.classifier(cls_repr)
        topic_input = self.grl(cls_repr)
        topic_logits = self.topic_head(topic_input)
        return logits, topic_logits


def main():
    from transformers import AutoTokenizer, AutoModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    MODEL_NAME = 'microsoft/deberta-v3-base'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_df = load_av_data(split='train')
    dev_df = load_av_data(split='dev')
    dev_labels = load_solution_labels(task='av')

    MAX_LEN, BATCH_SIZE, EPOCHS = 384, 8, 25
    PATIENCE, LR = 7, 2e-5
    NUM_TOPICS, GRL_LAMBDA, TOPIC_LOSS_WEIGHT = 10, 0.05, 0.1

    # Generate topic pseudo-labels via TF-IDF + KMeans
    print("Generating topic pseudo-labels for training data...")
    all_train_texts = list(train_df['text_1']) + list(train_df['text_2'])
    all_topic_labels = generate_topic_labels(all_train_texts, n_clusters=NUM_TOPICS)
    train_topic_labels = all_topic_labels[:len(train_df)]

    print("Generating topic pseudo-labels for dev data...")
    all_dev_texts = list(dev_df['text_1']) + list(dev_df['text_2'])
    dev_topic_all = generate_topic_labels(all_dev_texts, n_clusters=NUM_TOPICS)
    dev_topic_labels = dev_topic_all[:len(dev_df)]

    train_dataset = AVCrossEncoderDatasetV2(
        train_df, tokenizer, max_len=MAX_LEN, topic_labels=train_topic_labels)
    dev_dataset = AVCrossEncoderDatasetV2(
        dev_df, tokenizer, max_len=MAX_LEN, topic_labels=dev_topic_labels)
    dev_dataset.labels = np.array(dev_labels, dtype=np.float32)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    encoder = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model = AVCrossEncoderV2(encoder, num_topics=NUM_TOPICS, grl_lambda=GRL_LAMBDA).to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': LR},
        {'params': model.scalar_mix.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 5e-4},
        {'params': model.topic_head.parameters(), 'lr': 5e-4},
        {'params': model.grl.parameters(), 'lr': 0.0},
    ], weight_decay=0.01)
    scaler = GradScaler('cuda')

    best_f1, patience_counter = 0, 0
    save_dir = PROJECT_ROOT / 'models'
    save_dir.mkdir(exist_ok=True)

    print(f"\nTraining AV Cat C Cross-Encoder v2 (ScalarMix + GRL)")
    print(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")
    print(f"  Topics: {NUM_TOPICS}, GRL lambda: {GRL_LAMBDA}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_main, total_topic = 0, 0, 0
        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            topics = batch['topic'].to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                logits, topic_logits = model(ids, mask)
                main_loss = bce_loss(logits.squeeze(-1), labels)
                topic_loss = ce_loss(topic_logits, topics)
                loss = main_loss + TOPIC_LOSS_WEIGHT * topic_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            total_main += main_loss.item()
            total_topic += topic_loss.item()
