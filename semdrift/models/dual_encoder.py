"""
semdrift.models.dual_encoder — Dual Encoder PyTorch Model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class DualEncoderModel(nn.Module):
    """Dual-encoder model that processes code and docstring in separate forward passes."""

    def __init__(self, model_name: str, variant: str = "variant_2", freeze_base: bool = False):
        super().__init__()
        self.variant = variant
        self.encoder = AutoModel.from_pretrained(model_name)
        
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.variant == "variant_2":
            hidden_size = self.encoder.config.hidden_size
            self.classifier = nn.Linear(3 * hidden_size, 2)

    def mean_pooling(self, last_hidden_state, attention_mask):
        token_embeddings = last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(self, code_inputs, doc_inputs):
        code_outputs = self.encoder(**code_inputs)
        doc_outputs = self.encoder(**doc_inputs)

        code_emb = self.mean_pooling(code_outputs.last_hidden_state, code_inputs["attention_mask"])
        doc_emb = self.mean_pooling(doc_outputs.last_hidden_state, doc_inputs["attention_mask"])

        if self.variant == "variant_2":
            feat = torch.cat([code_emb, doc_emb, torch.abs(code_emb - doc_emb)], dim=1)
            logits = self.classifier(feat)
            return logits, code_emb, doc_emb
        else:
            return code_emb, doc_emb
