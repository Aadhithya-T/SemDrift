"""
semdrift.models.sliding_window — Sliding-Window Generation & Sequence Assembly Utilities.

Provides deterministic, deduplicated window generation with full tail coverage
and exact sequence assembly invariants for Joint Encoder CodeBERT architectures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer


def slice_docstring_windows(
    doc_ids: list[int],
    window_size: int = 192,
    stride: int = 128,
) -> list[list[int]]:
    """Create ordered, deduplicated windows with full tail coverage.

    Guarantees:
    1. First window starts at index 0.
    2. Consecutive strided windows step by `stride`.
    3. Terminal window always ends at len(doc_ids) (tail_start = len - window_size).
    4. Exact deduplication preserving insertion order.

    Args:
        doc_ids: List of token IDs (without special tokens).
        window_size: Maximum tokens per window (default 192).
        stride: Step size between window start indices (default 128).

    Returns:
        List of token ID lists, each of length <= window_size.
    """
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if stride > window_size:
        raise ValueError("stride cannot exceed window_size")

    n = len(doc_ids)
    if n <= window_size:
        return [doc_ids]

    starts = list(range(0, n - window_size + 1, stride))
    tail_start = n - window_size

    # Guarantee that the final window ends at the document tail.
    if starts[-1] != tail_start:
        starts.append(tail_start)

    # Preserve order and remove duplicate start positions.
    starts = list(dict.fromkeys(starts))

    return [
        doc_ids[start : start + window_size]
        for start in starts
    ]


def assemble_joint_sequence(
    doc_window_ids: list[int],
    code_ids: list[int],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    code_truncation: str = "head_tail",
) -> list[int]:
    """Assemble a single joint encoder sequence from doc window and code tokens.

    Format:
        [CLS] doc_window_tokens [SEP] [SEP] code_tokens [SEP]

    Guarantees:
    - 4 special tokens ([CLS], [SEP], [SEP], [SEP]) are properly accounted for.
    - Code is truncated using `code_truncation` (head_tail, head, or tail).
    - Resulting sequence length is strictly <= max_length.

    Args:
        doc_window_ids: Token IDs for the documentation window.
        code_ids: Token IDs for the code snippet.
        tokenizer: Pretrained tokenizer providing special token IDs.
        max_length: Maximum combined sequence length (default 512).
        code_truncation: Truncation strategy for code ('head_tail', 'head', 'tail').

    Returns:
        List of combined token IDs of length <= max_length.
    """
    cls_tok = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
    sep_tok = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 2
    mask_tok = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.unk_token_id

    # Bound doc window to ensure special tokens fit
    doc_window = doc_window_ids[: max_length - 4]
    remaining_budget = max_length - len(doc_window) - 4

    truncated_code = list(code_ids)
    if len(truncated_code) > remaining_budget:
        if remaining_budget <= 0:
            truncated_code = []
        elif code_truncation == "head_tail":
            code_budget = remaining_budget - 1
            if code_budget > 0:
                head_len = code_budget // 2
                tail_len = code_budget - head_len
                truncated_code = truncated_code[:head_len] + [mask_tok] + truncated_code[-tail_len:]
            else:
                truncated_code = [mask_tok]
        elif code_truncation == "tail":
            truncated_code = truncated_code[-remaining_budget:]
        else:  # head or fallback
            truncated_code = truncated_code[:remaining_budget]

    seq = [cls_tok] + doc_window + [sep_tok, sep_tok] + truncated_code + [sep_tok]

    # HARD INVARIANT: strictly <= max_length
    assert len(seq) <= max_length, f"Assembled sequence length {len(seq)} exceeds max_length {max_length}"
    return seq


def build_function_sliding_windows(
    docstring: str,
    code: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    window_size: int = 192,
    stride: int = 128,
    code_truncation: str = "head_tail",
) -> list[list[int]]:
    """Tokenize a (docstring, code) pair and generate all sliding-window joint sequences.

    Returns:
        List of assembled token ID sequences (each of length <= max_length).
    """
    doc_ids = tokenizer.encode(docstring, add_special_tokens=False)
    code_ids = tokenizer.encode(code, add_special_tokens=False)

    doc_windows = slice_docstring_windows(doc_ids, window_size=window_size, stride=stride)

    sequences = []
    for win in doc_windows:
        seq = assemble_joint_sequence(
            doc_window_ids=win,
            code_ids=code_ids,
            tokenizer=tokenizer,
            max_length=max_length,
            code_truncation=code_truncation,
        )
        sequences.append(seq)

    return sequences
