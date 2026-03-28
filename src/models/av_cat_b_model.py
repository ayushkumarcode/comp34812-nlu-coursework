"""
Cat B model: Siamese char-CNN + BiLSTM + attention with GRL for
topic adversarial debiasing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """GRL from Ganin & Lempitsky (2015)."""

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val=0.1):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


class AdditiveAttention(nn.Module):
    """Bahdanau attention -- learns which timesteps matter most."""

    def __init__(self, hidden_size, attention_size=128):
        super().__init__()
        self.W_a = nn.Linear(hidden_size, attention_size)
        self.v_a = nn.Linear(attention_size, 1, bias=False)

    def forward(self, lstm_output, mask=None):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
            mask: (batch, seq_len) - True for valid positions

        Returns:
            attended: (batch, hidden_size)
            weights: (batch, seq_len)
        """
        scores = self.v_a(torch.tanh(self.W_a(lstm_output)))  # (batch, seq, 1)
        scores = scores.squeeze(-1)  # (batch, seq)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        weights = F.softmax(scores, dim=1)  # (batch, seq)
        attended = (weights.unsqueeze(-1) * lstm_output).sum(dim=1)  # (batch, hidden)
        return attended, weights


# ============================================================
# SHARED ENCODER (SIAMESE)
# ============================================================

class SharedEncoder(nn.Module):
    """Character-level CNN + BiLSTM + Attention encoder."""

    def __init__(self, vocab_size=97, char_emb_dim=32,
                 cnn_filters=128, lstm_hidden=128,
                 proj_dim=128, dropout=0.2):
        super().__init__()

        self.char_emb = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)

        # Multi-width Conv1D
        self.conv3 = nn.Conv1d(char_emb_dim, cnn_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(char_emb_dim, cnn_filters, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(char_emb_dim, cnn_filters, kernel_size=7, padding=3)

        total_filters = cnn_filters * 3  # 384
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.cnn_dropout = nn.Dropout(dropout)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=total_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        lstm_output_size = lstm_hidden * 2  # 256

        # Attention
        self.attention = AdditiveAttention(lstm_output_size, 128)

        # Projection
        self.proj = nn.Sequential(
            nn.Linear(lstm_output_size, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, char_ids):
        """
        Args:
            char_ids: (batch, seq_len) character indices

        Returns:
            embedding: (batch, proj_dim)
            attention_weights: (batch, seq_len // 3)
        """
        # Character embedding
        x = self.char_emb(char_ids)  # (batch, seq, emb_dim)
        x = x.permute(0, 2, 1)  # (batch, emb_dim, seq)

        # Multi-width CNN
        c3 = F.relu(self.conv3(x))
        c5 = F.relu(self.conv5(x))
        c7 = F.relu(self.conv7(x))
        x = torch.cat([c3, c5, c7], dim=1)  # (batch, 384, seq)

        # Pool and dropout
        x = self.pool(x)  # (batch, 384, seq//3)
        x = self.cnn_dropout(x)
        x = x.permute(0, 2, 1)  # (batch, seq//3, 384)

        # BiLSTM
        x, _ = self.lstm(x)  # (batch, seq//3, 256)

        # Attention
        attended, weights = self.attention(x)  # (batch, 256)

        # Projection
        embedding = self.proj(attended)  # (batch, 128)

        return embedding, weights


# ============================================================
# FULL MODEL
# ============================================================

class AVCatBModel(nn.Module):
    """Adversarial Style-Content Disentanglement Network."""

    def __init__(self, vocab_size=97, char_emb_dim=32,
                 cnn_filters=128, lstm_hidden=128,
                 proj_dim=128, num_topics=10,
                 grl_lambda=0.1, dropout=0.2):
        super().__init__()

        self.encoder = SharedEncoder(
            vocab_size=vocab_size,
            char_emb_dim=char_emb_dim,
            cnn_filters=cnn_filters,
            lstm_hidden=lstm_hidden,
            proj_dim=proj_dim,
            dropout=dropout,
        )

        # Comparison + MLP Classifier
        comparison_dim = proj_dim * 4  # [v1, v2, |v1-v2|, v1*v2] = 512
        self.classifier = nn.Sequential(
            nn.Linear(comparison_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        # Topic adversarial head
        self.grl = GradientReversalLayer(lambda_val=grl_lambda)
        self.topic_head = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_topics),
        )

    def forward(self, char_ids_1, char_ids_2, return_embeddings=False):
        """
        Args:
            char_ids_1: (batch, seq_len) text 1 character indices
            char_ids_2: (batch, seq_len) text 2 character indices

        Returns:
            logits: (batch, 1) verification logits
            topic_logits: (batch, num_topics) topic prediction
            embeddings: (v1, v2) if return_embeddings
        """
        v1, attn1 = self.encoder(char_ids_1)
        v2, attn2 = self.encoder(char_ids_2)

        # Comparison
        diff = torch.abs(v1 - v2)
        prod = v1 * v2
        combined = torch.cat([v1, v2, diff, prod], dim=1)

        # Classification
        logits = self.classifier(combined)

        # Topic prediction (on randomly selected text embedding)
        # Use v1 for simplicity (alternating in training loop)
        topic_input = self.grl(v1)
        topic_logits = self.topic_head(topic_input)

        if return_embeddings:
            return logits, topic_logits, (v1, v2, attn1, attn2)

        return logits, topic_logits

    def predict(self, char_ids_1, char_ids_2):
        """Generate binary predictions."""
        logits, _ = self.forward(char_ids_1, char_ids_2)
        probs = torch.sigmoid(logits)
        return (probs > 0.5).long().squeeze(-1)
