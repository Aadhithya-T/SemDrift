"""
semdrift.models.joint_encoder — Joint Encoder PyTorch Model & Dataset Utilities.
"""

from __future__ import annotations

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


def extract_docstring_summary(docstring: str) -> str:
    """Extract clean natural language summary from docstrings.

    Strips REPL examples (>>>), parameter tables, return specs, etc.
    Keeps only the opening summary sentence(s).
    """
    if not docstring:
        return ""
    lines = docstring.strip().split("\n")
    summary_lines = []
    for line in lines:
        stripped = line.strip()
        if (not stripped
                or stripped.startswith(">>>")
                or stripped.startswith("...")
                or stripped.startswith("Parameters")
                or stripped.startswith("Returns")
                or stripped.startswith("Examples")
                or stripped.startswith("See Also")
                or stripped.startswith("Notes")
                or stripped.startswith("Raises")
                or stripped.startswith("Warnings")
                or stripped.startswith("References")):
            break
        summary_lines.append(stripped)

    cleaned = " ".join(summary_lines).strip()
    if len(cleaned) >= 10:
        return cleaned
    return lines[0].strip()


class SemDriftDataset(Dataset):
    """Loads JSONL records with fields: code, docstring, label, drift_type, severity."""

    def __init__(self, filepath: str, clean_docs: bool = True):
        self.records: list[dict] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if clean_docs:
                        rec["docstring"] = extract_docstring_summary(rec.get("docstring", ""))
                    self.records.append(rec)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        code = rec.get("code", "")
        docstring = rec.get("docstring", "")
        label_str = rec.get("label", "aligned")
        label = 1 if label_str == "drifted" else 0

        meta = {
            "repo": rec.get("repo", "unknown"),
            "drift_type": rec.get("drift_type") or "aligned",
            "severity": rec.get("severity") or "aligned",
            "label_str": label_str,
        }
        return docstring, code, label, meta


def make_collate_fn(tokenizer: AutoTokenizer, max_length: int, doc_max_tokens: int, truncation_strategy: str):
    """Create a collate function that tokenizes (docstring, code) as a SINGLE
    pair input, applying custom truncation to prevent information loss at the tail."""

    def collate_fn(batch):
        docstrings = [item[0] for item in batch]
        codes = [item[1] for item in batch]
        labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
        metas = [item[3] for item in batch]

        input_ids_list = []
        mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.unk_token_id
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        for doc, code in zip(docstrings, codes):
            doc_ids = tokenizer.encode(doc, add_special_tokens=False)
            code_ids = tokenizer.encode(code, add_special_tokens=False)

            doc_ids = doc_ids[:doc_max_tokens]
            remaining_budget = max_length - len(doc_ids) - 4

            if len(code_ids) > remaining_budget:
                if truncation_strategy == "head_tail":
                    code_budget = remaining_budget - 1
                    if code_budget > 0:
                        head_len = code_budget // 2
                        tail_len = code_budget - head_len
                        code_ids = code_ids[:head_len] + [mask_token_id] + code_ids[-tail_len:]
                    else:
                        code_ids = [mask_token_id]
                elif truncation_strategy == "head":
                    code_ids = code_ids[-remaining_budget:]
                else:
                    code_ids = code_ids[:remaining_budget]

            cls_tok = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
            sep_tok = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 2
            combined_ids = [cls_tok] + doc_ids + [sep_tok, sep_tok] + code_ids + [sep_tok]
            combined_ids = combined_ids[:max_length]
            
            input_ids_list.append(combined_ids)

        max_batch_len = max(len(ids) for ids in input_ids_list)
        padded_input_ids = []
        padded_attention_mask = []

        for ids in input_ids_list:
            pad_len = max_batch_len - len(ids)
            padded_input_ids.append(ids + [pad_token_id] * pad_len)
            padded_attention_mask.append([1] * len(ids) + [0] * pad_len)

        inputs = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long)
        }
        return inputs, labels, metas

    return collate_fn


class JointEncoderModel(nn.Module):
    """Joint encoder for semantic drift detection with customizable pooling."""

    def __init__(self, model_name: str, pooling: str = "cls", num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def mean_pooling(self, last_hidden_state, attention_mask):
        token_embeddings = last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(self, inputs: dict) -> torch.Tensor:
        outputs = self.encoder(**inputs)

        if self.pooling == "cls":
            pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        else:
            pooled = self.mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class FocalLoss(nn.Module):
    """Focal Loss to focus training gradients on hard, low-confidence examples (e.g., doc_negation)."""

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sample_weights: torch.Tensor = None) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss

        if sample_weights is not None:
            focal_loss = focal_loss * sample_weights

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
