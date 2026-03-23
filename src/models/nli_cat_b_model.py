"""
NLI Category B — ESIM with KIM-inspired WordNet Knowledge Enhancement.
Enhanced Sequential Inference Model + Character CNN + Knowledge Injection.

Architecture:
  Input Encoding: Word Embedding + Char CNN -> BiLSTM
  Cross-Attention: Soft alignment between premise and hypothesis
  Knowledge Enhancement: WordNet relation features injected at comparison
  Enhancement: [orig; aligned; diff; product; knowledge]
  Composition: BiLSTM
  Pooling: Avg + Max pooling
  Classification: MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    """Character-level CNN for morphological word representations."""

    def __init__(self, char_vocab_size=98, char_emb_dim=8,
                 num_filters=50, kernel_size=5):
        super().__init__()
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.conv = nn.Conv1d(char_emb_dim, num_filters, kernel_size, padding=kernel_size//2)

    def forward(self, char_ids):
        """
        Args:
            char_ids: (batch, seq_len, char_len)

        Returns:
            char_repr: (batch, seq_len, num_filters)
        """
        batch, seq_len, char_len = char_ids.shape

        # Reshape for conv
        x = char_ids.view(batch * seq_len, char_len)
        x = self.char_emb(x)  # (batch*seq, char_len, emb)
        x = x.permute(0, 2, 1)  # (batch*seq, emb, char_len)
        x = F.relu(self.conv(x))  # (batch*seq, filters, char_len)
        x = x.max(dim=2)[0]  # (batch*seq, filters) max pool over chars

        return x.view(batch, seq_len, -1)


class ESIM(nn.Module):
    """Enhanced Sequential Inference Model with Knowledge Enhancement."""

    def __init__(self, vocab_size, embedding_dim=300, hidden_size=300,
                 char_vocab_size=98, char_emb_dim=8, char_filters=50,
                 knowledge_dim=5, dropout=0.3, pretrained_embeddings=None):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Word embedding
        self.word_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.word_emb.weight.data.copy_(pretrained_embeddings)
            self.word_emb.weight.requires_grad = False  # Frozen initially

        # Character CNN
        self.char_cnn = CharCNN(char_vocab_size, char_emb_dim, char_filters)

        # Input projection
        input_dim = embedding_dim + char_filters
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
        )

        # Input encoding BiLSTM
        self.input_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Knowledge projection
        self.knowledge_proj = nn.Sequential(
            nn.Linear(knowledge_dim, 50),
            nn.ReLU(),
        )

        # Enhancement projection
        # [orig (2H); aligned (2H); diff (2H); product (2H); knowledge (50)]
        enhancement_dim = hidden_size * 2 * 4 + 50
        self.enhance_proj = nn.Sequential(
            nn.Linear(enhancement_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Composition BiLSTM
        self.composition_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Classification MLP
        # [p_avg; p_max; h_avg; h_max] = 4 * 2H
        mlp_input = hidden_size * 2 * 4
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input, 512),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, premise_word_ids, premise_char_ids,
                hypothesis_word_ids, hypothesis_char_ids,
                wordnet_relations=None):
        """
        Args:
            premise_word_ids: (batch, p_len)
            premise_char_ids: (batch, p_len, char_len)
            hypothesis_word_ids: (batch, h_len)
            hypothesis_char_ids: (batch, h_len, char_len)
            wordnet_relations: (batch, p_len, h_len, 5) or None

        Returns:
            logits: (batch, 1)
        """
        # Create masks
        p_mask = (premise_word_ids != 0)  # (batch, p_len)
        h_mask = (hypothesis_word_ids != 0)  # (batch, h_len)

        # ===== INPUT ENCODING =====
        # Word embeddings
        p_word = self.word_emb(premise_word_ids)  # (batch, p_len, emb)
        h_word = self.word_emb(hypothesis_word_ids)

        # Char CNN
        p_char = self.char_cnn(premise_char_ids)  # (batch, p_len, char_filters)
        h_char = self.char_cnn(hypothesis_char_ids)

        # Concatenate word + char
        p_emb = torch.cat([p_word, p_char], dim=2)
        h_emb = torch.cat([h_word, h_char], dim=2)

        # Project
        p_proj = self.input_proj(p_emb)
        h_proj = self.input_proj(h_emb)

        # Input BiLSTM
        p_enc, _ = self.input_lstm(p_proj)  # (batch, p_len, 2H)
        h_enc, _ = self.input_lstm(h_proj)  # (batch, h_len, 2H)

        # ===== CROSS-ATTENTION =====
        # Attention matrix
        attn = torch.bmm(p_enc, h_enc.transpose(1, 2))  # (batch, p_len, h_len)

        # Mask attention
        p_mask_exp = p_mask.unsqueeze(2).float()  # (batch, p_len, 1)
        h_mask_exp = h_mask.unsqueeze(1).float()  # (batch, 1, h_len)
        attn_mask = p_mask_exp * h_mask_exp
        attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        # Soft alignment (handle all-masked rows to prevent NaN)
        p_attn_weights = F.softmax(attn, dim=2)  # attend to hypothesis
        p_attn_weights = p_attn_weights.nan_to_num(0.0)
        h_attn_weights = F.softmax(attn.transpose(1, 2), dim=2)  # attend to premise
        h_attn_weights = h_attn_weights.nan_to_num(0.0)

        p_aligned = torch.bmm(p_attn_weights, h_enc)  # (batch, p_len, 2H)
        h_aligned = torch.bmm(h_attn_weights, p_enc)  # (batch, h_len, 2H)

        # ===== KNOWLEDGE ENHANCEMENT =====
        if wordnet_relations is not None:
            # Weight relation vectors by attention
            # wordnet_relations: (batch, p_len, h_len, 5)
            # p_attn_weights: (batch, p_len, h_len)
            k_p = torch.einsum('bph,bphk->bpk',
                               p_attn_weights, wordnet_relations.float())
            k_h = torch.einsum('bhp,bphk->bhk',
                               h_attn_weights, wordnet_relations.float())

            k_p_proj = self.knowledge_proj(k_p)  # (batch, p_len, 50)
            k_h_proj = self.knowledge_proj(k_h)  # (batch, h_len, 50)
        else:
            k_p_proj = torch.zeros(p_enc.shape[0], p_enc.shape[1], 50,
                                   device=p_enc.device)
            k_h_proj = torch.zeros(h_enc.shape[0], h_enc.shape[1], 50,
                                   device=h_enc.device)

        # ===== ENHANCEMENT =====
        m_p = torch.cat([
            p_enc, p_aligned, p_enc - p_aligned, p_enc * p_aligned, k_p_proj
        ], dim=2)
        m_h = torch.cat([
            h_enc, h_aligned, h_enc - h_aligned, h_enc * h_aligned, k_h_proj
        ], dim=2)

        m_p = self.enhance_proj(m_p)  # (batch, p_len, H)
        m_h = self.enhance_proj(m_h)  # (batch, h_len, H)

        # ===== COMPOSITION =====
        v_p, _ = self.composition_lstm(m_p)  # (batch, p_len, 2H)
        v_h, _ = self.composition_lstm(m_h)  # (batch, h_len, 2H)

        # ===== POOLING =====
        # Masked average and max pooling
        p_avg = self._masked_avg_pool(v_p, p_mask)
        p_max = self._masked_max_pool(v_p, p_mask)
        h_avg = self._masked_avg_pool(v_h, h_mask)
        h_max = self._masked_max_pool(v_h, h_mask)

        v = torch.cat([p_avg, p_max, h_avg, h_max], dim=1)

        # ===== CLASSIFICATION =====
        logits = self.mlp(v)
        return logits

    def _masked_avg_pool(self, x, mask):
        """Average pooling with mask."""
        mask_f = mask.unsqueeze(-1).float()
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

    def _masked_max_pool(self, x, mask):
        """Max pooling with mask."""
        mask_f = mask.unsqueeze(-1).float()
        x = x.masked_fill(mask_f == 0, float('-inf'))
        return x.max(dim=1)[0]

    def unfreeze_embeddings(self):
        """Unfreeze word embeddings for fine-tuning."""
        self.word_emb.weight.requires_grad = True
