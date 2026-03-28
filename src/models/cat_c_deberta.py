"""
Cat C DeBERTa models. AV uses a siamese setup with layer-weighted CLS,
NLI uses a cross-encoder with hypothesis-only adversarial debiasing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GRL(nn.Module):
    def __init__(self, lambda_val=0.05):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


class ScalarMix(nn.Module):
    """Learnable scalar mixture of layer outputs (a la ELMo, Peters et al. 2018)."""

    def __init__(self, num_layers=12, style_bias=True):
        super().__init__()
        if style_bias:
            init_weights = torch.tensor(
                [0.12] * 4 + [0.07] * 8, dtype=torch.float
            )
        else:
            init_weights = torch.ones(num_layers) / num_layers
        self.weights = nn.Parameter(init_weights)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: List of (batch, seq_len, hidden) tensors.

        Returns:
            Weighted sum: (batch, seq_len, hidden)
        """
        normed = F.softmax(self.weights, dim=0)
        mixed = sum(w * h for w, h in zip(normed, hidden_states))
        return mixed


class AVDeBERTaSiamese(nn.Module):
    """Siamese DeBERTa for AV. Each text gets its own [CLS] embedding,
    then we compare them with an MLP."""

    def __init__(self, model_name='microsoft/deberta-v3-base',
                 proj_dim=128, num_topics=10, grl_lambda=0.05):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        )
        hidden_size = self.encoder.config.hidden_size  # 768
        num_layers = self.encoder.config.num_hidden_layers  # 12

        self.scalar_mix = ScalarMix(num_layers, style_bias=True)

        self.style_proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, proj_dim),
        )

        comparison_dim = proj_dim * 4 + 1
        self.classifier = nn.Sequential(
            nn.Linear(comparison_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        self.grl = GRL(grl_lambda)
        self.topic_head = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_topics),
        )

    def encode(self, input_ids, attention_mask):
        """Encode one text to a normalized style embedding."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.hidden_states[1:]  # 12 layers

        cls_states = [h[:, 0, :] for h in hidden_states]  # each (batch, hidden)
        stacked = torch.stack(cls_states, dim=0)  # (layers, batch, hidden)
        mixed_cls = sum(
            w * s for w, s in zip(
                F.softmax(self.scalar_mix.weights, dim=0), cls_states
            )
        )

        style_emb = self.style_proj(mixed_cls)
        style_emb = F.normalize(style_emb, p=2, dim=1)

        return style_emb

    def forward(self, input_ids_1, attention_mask_1,
                input_ids_2, attention_mask_2):
        """
        Returns:
            logits: (batch, 1)
            topic_logits: (batch, num_topics)
            embeddings: (v1, v2)
        """
        v1 = self.encode(input_ids_1, attention_mask_1)
        v2 = self.encode(input_ids_2, attention_mask_2)

        diff = torch.abs(v1 - v2)
        prod = v1 * v2
        cos_sim = (v1 * v2).sum(dim=1, keepdim=True)
        combined = torch.cat([v1, v2, diff, prod, cos_sim], dim=1)

        logits = self.classifier(combined)

        topic_input = self.grl(v1)
        topic_logits = self.topic_head(topic_input)

        return logits, topic_logits, (v1, v2)


class NLIDeBERTaCrossEncoder(nn.Module):
    """NLI cross-encoder with GRL on hypothesis-only representation
    to prevent the model from relying on hypothesis artifacts."""

    def __init__(self, model_name='microsoft/deberta-v3-base',
                 grl_lambda=0.1):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

        self.hyp_encoder = AutoModel.from_pretrained(model_name)

        self.grl = GRL(grl_lambda)
        self.adversarial_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask,
                hyp_input_ids=None, hyp_attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len) concatenated premise+hypothesis
            attention_mask: (batch, seq_len)
            hyp_input_ids: (batch, hyp_len) hypothesis-only
            hyp_attention_mask: (batch, hyp_len)

        Returns:
            logits: (batch, 1)
            adv_logits: (batch, 1) or None
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_repr)

        adv_logits = None
        if hyp_input_ids is not None:
            hyp_outputs = self.hyp_encoder(
                input_ids=hyp_input_ids, attention_mask=hyp_attention_mask
            )
            hyp_cls = hyp_outputs.last_hidden_state[:, 0, :]
            adv_input = self.grl(hyp_cls)
            adv_logits = self.adversarial_head(adv_input)

        return logits, adv_logits
