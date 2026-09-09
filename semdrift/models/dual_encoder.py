"""
semdrift.models.dual_encoder — Dual Encoder PyTorch Model & Dataset Utilities.
"""

from __future__ import annotations

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

from semdrift.data.labels import extract_label_tuple


def extract_docstring_summary(docstring: str) -> str:
    """Extract clean natural language summary from docstrings."""
    if not docstring:
        return ""
    lines = docstring.strip().split("\n")
    summary_lines = []
    for line in lines:
        l = line.strip()
        if not l or l.startswith(">>>") or l.startswith("...") or l.startswith("Parameters") or l.startswith("Returns") or l.startswith("Examples") or l.startswith("See Also"):
            break
        summary_lines.append(l)

    cleaned = " ".join(summary_lines).strip()
    if len(cleaned) >= 10:
        return cleaned
    return lines[0].strip()


class DualEncoderDataset(Dataset):
    """Dataset for Dual-Encoder model (returns code, docstring, label, meta) across V1 and V2 schemas."""

    def __init__(self, filepath: str, clean_docs: bool = True):
        self.records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    raw_code = rec.get("code") or rec.get("code_after") or rec.get("code_before") or ""
                    raw_doc = rec.get("docstring") or rec.get("docstring_after") or rec.get("docstring_before") or ""

                    if clean_docs:
                        rec["docstring"] = extract_docstring_summary(raw_doc)
                    else:
                        rec["docstring"] = raw_doc
                    rec["code"] = raw_code

                    rec["label_int"], rec["label_str"] = self._extract_label(rec)
                    self.records.append(rec)

    @staticmethod
    def _extract_label(rec: dict) -> tuple[int, str]:
        """Extract canonical label (int, str) using authoritative semdrift.data.labels helper."""
        return extract_label_tuple(rec)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        code = rec["code"]
        docstring = rec["docstring"]
        label = rec["label_int"]
        label_str = rec["label_str"]

        # Meta dictionary for evaluating breakdowns later
        meta = {
            "repo": rec.get("repo") or rec.get("repo_name") or rec.get("repo_or_origin") or "unknown",
            "drift_type": rec.get("drift_type") or rec.get("curation_method") or rec.get("drift_source") or label_str,
            "provenance": rec.get("provenance") or rec.get("source") or ("contract_grounded_generated" if label == 1 else "clean_grounded"),
            "severity": rec.get("severity") or ("aligned" if label == 0 else "unknown"),
            "label_str": label_str,
        }
        return code, docstring, label, meta


# Alias for backwards compatibility
SemDriftDataset = DualEncoderDataset


def make_collate_fn(tokenizer: AutoTokenizer, max_length: int):
    """Create a collate function that tokenizes code and docstring separately."""
    def collate_fn(batch):
        codes = [item[0] for item in batch]
        docstrings = [item[1] for item in batch]
        labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
        metas = [item[3] for item in batch]

        # Tokenize code and docstring separately (isolated encoding without joint self-attention)
        code_inputs = tokenizer(
            codes,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        doc_inputs = tokenizer(
            docstrings,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return code_inputs, doc_inputs, labels, metas
    return collate_fn


class DualEncoderModel(nn.Module):
    """Dual-encoder model that processes code and docstring in separate forward passes."""

    def __init__(self, model_name: str, variant: str = "variant_2", freeze_base: bool = False, dropout: float = 0.1):
        super().__init__()
        self.variant = variant
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        if freeze_base:
            print("Freezing base model layers. Fine-tuning only the classifier head.")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            print("Fine-tuning base model + classification/projection layers end-to-end.")

        if self.variant == "variant_2":
            # Classifier head on top of [u; v; |u-v|]
            hidden_size = self.encoder.config.hidden_size
            self.classifier = nn.Linear(3 * hidden_size, 2)

    def mean_pooling(self, last_hidden_state, attention_mask):
        token_embeddings = last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(self, code_inputs, doc_inputs):
        # Step 1: Independently encode code and docstrings
        code_outputs = self.encoder(**code_inputs)
        doc_outputs = self.encoder(**doc_inputs)

        # Step 2: Mean-pool over all token representations (excluding padding tokens)
        code_emb = self.mean_pooling(code_outputs.last_hidden_state, code_inputs["attention_mask"])
        doc_emb = self.mean_pooling(doc_outputs.last_hidden_state, doc_inputs["attention_mask"])

        if self.variant == "variant_2":
            # Concatenate u, v, and |u - v|
            feat = torch.cat([code_emb, doc_emb, torch.abs(code_emb - doc_emb)], dim=1)
            feat = self.dropout(feat)
            logits = self.classifier(feat)
            return logits, code_emb, doc_emb
        else:
            # Variant 1: return representations directly for distance/similarity training
            return code_emb, doc_emb

