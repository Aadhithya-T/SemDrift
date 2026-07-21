"""
semdrift.embedder.embed — CodeBERT embedding and divergence scoring (Model A Baseline).

Provides:
    - ``get_embeddings(texts, batch_size, pooling, device)`` — batched embeddings
      (mean-pooled or [CLS]-pooled) for an arbitrary list of strings.
    - ``embed_function_record(record, ...)`` — embeds ``code`` and ``docstring``
      from a parser-output or dataset record.
    - ``compute_divergence(code_emb, doc_emb, normalize=False)`` — divergence computation.
    - ``predict_drift(divergence, threshold)`` — string label ("drifted" / "aligned").
"""

from __future__ import annotations

import os
import sys
import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
import yaml
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default GPU if CUDA is available, else CPU
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Config helper & Module-level model cache
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")

def load_config(config_path: str = _CONFIG_PATH) -> dict:
    """Load configuration from config.yaml if it exists, else return default config."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


_tokenizer: AutoTokenizer | None = None
_model: AutoModel | None = None
_cached_model_name: str | None = None
_cached_device: str | None = None


def _ensure_model_loaded(
    model_name: str = "microsoft/codebert-base",
    device: str = DEFAULT_DEVICE,
) -> tuple[AutoTokenizer, AutoModel, str]:
    """Lazily load tokenizer and model on first use or when device/model changes."""
    global _tokenizer, _model, _cached_model_name, _cached_device  # noqa: PLW0603

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    if _tokenizer is None or _model is None or _cached_model_name != model_name or _cached_device != device:
        logger.info("Loading tokenizer for '%s' …", model_name)
        _tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info("Loading model for '%s' on device '%s' …", model_name, device)
        _model = AutoModel.from_pretrained(model_name)
        _model.to(device)
        _model.eval()  # inference-only

        _cached_model_name = model_name
        _cached_device = device
        logger.info("Model loaded successfully on %s.", device.upper())

    return _tokenizer, _model, device


# ---------------------------------------------------------------------------
# Core embedding function
# ---------------------------------------------------------------------------

def get_embeddings(
    texts: list[str],
    batch_size: int = 64,
    max_token_length: int = 512,
    model_name: str = "microsoft/codebert-base",
    device: str = DEFAULT_DEVICE,
    pooling: str = "mean",
    show_progress: bool = False,
) -> torch.Tensor:
    """Embed a list of texts using CodeBERT (supports CUDA GPU acceleration)."""
    if not texts:
        return torch.empty((0, 768))

    tokenizer, model, target_device = _ensure_model_loaded(model_name=model_name, device=device)

    all_embeddings: list[torch.Tensor] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i, start in enumerate(range(0, len(texts), batch_size)):
        batch_texts = texts[start : start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_token_length,
            return_tensors="pt",
        )

        encoded = {k: v.to(target_device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        if pooling == "mean":
            attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)
            token_embeddings = outputs.last_hidden_state              # (batch, seq_len, hidden_dim)
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask
        elif pooling == "cls":
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling strategy: '{pooling}'. Use 'mean' or 'cls'.")

        all_embeddings.append(batch_embeddings.cpu())

        if show_progress and ((i + 1) % 5 == 0 or (i + 1) == total_batches):
            print(f"    Batch {i + 1}/{total_batches} ({min(start + batch_size, len(texts))}/{len(texts)} texts)", flush=True)

    return torch.cat(all_embeddings, dim=0)


# ---------------------------------------------------------------------------
# Record & Divergence Helpers
# ---------------------------------------------------------------------------

def embed_function_record(
    record: dict,
    batch_size: int = 64,
    max_token_length: int = 512,
    model_name: str = "microsoft/codebert-base",
    device: str = DEFAULT_DEVICE,
    pooling: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed the ``code`` and ``docstring`` from a single parser or dataset record."""
    code: str = record["code"]
    docstring: str = record["docstring"]

    embeddings = get_embeddings(
        [code, docstring],
        batch_size=batch_size,
        max_token_length=max_token_length,
        model_name=model_name,
        device=device,
        pooling=pooling,
    )

    return embeddings[0:1], embeddings[1:2]


def compute_divergence(
    code_emb: torch.Tensor,
    doc_emb: torch.Tensor,
    normalize: bool = False,
    mean_center: bool = False,
    code_mean: torch.Tensor | None = None,
    doc_mean: torch.Tensor | None = None,
) -> float | torch.Tensor:
    """Compute semantic divergence: ``1 - cosine_similarity(code_emb, doc_emb)``.

    When ``mean_center=True``, subtracts the dataset mean vector from embeddings
    to mitigate CodeBERT anisotropy before computing cosine similarity.
    When ``normalize=True``, applies L2-normalization across feature dimensions.
    """
    if code_emb.ndim == 1:
        code_emb = code_emb.unsqueeze(0)
    if doc_emb.ndim == 1:
        doc_emb = doc_emb.unsqueeze(0)

    if mean_center:
        if code_mean is not None:
            code_emb = code_emb - code_mean
        elif code_emb.size(0) > 1:
            code_emb = code_emb - code_emb.mean(dim=0, keepdim=True)

        if doc_mean is not None:
            doc_emb = doc_emb - doc_mean
        elif doc_emb.size(0) > 1:
            doc_emb = doc_emb - doc_emb.mean(dim=0, keepdim=True)

    if normalize:
        code_emb = F.normalize(code_emb, p=2, dim=1)
        doc_emb = F.normalize(doc_emb, p=2, dim=1)

    similarity = cosine_similarity(code_emb, doc_emb, dim=1)
    divergence = 1.0 - similarity

    if divergence.numel() == 1:
        return divergence.item()
    return divergence


def predict_drift(divergence: float, threshold: float = 0.15) -> str:
    """Classify divergence score into string label."""
    return "drifted" if divergence >= threshold else "aligned"