"""
tests/test_sliding_window.py — Unit Tests for Sliding-Window Slicing & Sequence Assembly.

Verifies:
- Document lengths 45, 192, 320, 384, and 800.
- Exact window start positions and window counts.
- Complete tail coverage (last window always ends at doc length).
- Strict deduplication (no redundant start indices or identical windows).
- Sequence assembly <= 512 invariant across diverse code and doc lengths.
- Parity of Window 0 with baseline collate function.
"""

import pytest
from transformers import AutoTokenizer

from semdrift.models.sliding_window import (
    slice_docstring_windows,
    assemble_joint_sequence,
    build_function_sliding_windows,
)
from semdrift.models.joint_encoder import make_collate_fn


class DummyTokenizer:
    """Lightweight dummy tokenizer for testing without downloading full pretrained models."""

    def __init__(self):
        self.cls_token_id = 0
        self.sep_token_id = 2
        self.mask_token_id = 50264
        self.pad_token_id = 1
        self.unk_token_id = 3

    def encode(self, text: str, add_special_tokens: bool = False):
        # Deterministic dummy mapping based on split words
        words = text.strip().split()
        return [hash(w) % 30000 + 10 for w in words]


# ============================================================================
# 1. Exact Window Slicing Boundary Tests (45, 192, 320, 384, 800)
# ============================================================================

@pytest.mark.parametrize(
    "doc_len, expected_count, expected_starts",
    [
        (45, 1, [0]),
        (192, 1, [0]),
        (320, 2, [0, 128]),
        (384, 3, [0, 128, 192]),
        (800, 6, [0, 128, 256, 384, 512, 608]),
    ],
)
def test_sliding_window_exact_boundaries(doc_len, expected_count, expected_starts):
    """Verify window count, exact start indices, deduplication, and tail coverage."""
    doc_ids = list(range(doc_len))
    windows = slice_docstring_windows(doc_ids, window_size=192, stride=128)

    # 1. Verify window count
    assert len(windows) == expected_count, (
        f"Length {doc_len}: expected {expected_count} windows, got {len(windows)}"
    )

    # 2. Verify exact start indices
    actual_starts = [w[0] for w in windows]
    assert actual_starts == expected_starts, (
        f"Length {doc_len}: expected starts {expected_starts}, got {actual_starts}"
    )

    # 3. Verify deduplication
    assert len(set(tuple(w) for w in windows)) == expected_count, (
        f"Length {doc_len}: duplicate windows detected!"
    )

    # 4. Verify tail coverage (last window must reach the final token)
    assert windows[-1][-1] == doc_len - 1, (
        f"Length {doc_len}: final window did not end at tail (ended at {windows[-1][-1]})"
    )

    # 5. Verify window size bound
    for i, w in enumerate(windows):
        if doc_len >= 192:
            assert len(w) == 192, f"Window {i} for length {doc_len} has length {len(w)} != 192"
        else:
            assert len(w) == doc_len


def test_invalid_parameters():
    """Verify parameter validation."""
    doc_ids = list(range(100))
    with pytest.raises(ValueError, match="positive"):
        slice_docstring_windows(doc_ids, window_size=0, stride=128)
    with pytest.raises(ValueError, match="positive"):
        slice_docstring_windows(doc_ids, window_size=192, stride=-10)
    with pytest.raises(ValueError, match="stride cannot exceed window_size"):
        slice_docstring_windows(doc_ids, window_size=128, stride=192)


# ============================================================================
# 2. Sequence Assembly & Invariants (<= 512 tokens)
# ============================================================================

@pytest.mark.parametrize("doc_len", [0, 45, 192, 320, 384, 800])
@pytest.mark.parametrize("code_len", [0, 10, 100, 316, 500, 2000])
@pytest.mark.parametrize("truncation", ["head_tail", "head", "tail"])
def test_assemble_joint_sequence_invariant(doc_len, code_len, truncation):
    """Ensure assembled sequence is strictly <= 512 tokens under all conditions."""
    tokenizer = DummyTokenizer()
    doc_ids = list(range(doc_len))
    code_ids = [c + 1000 for c in range(code_len)]

    windows = slice_docstring_windows(doc_ids, window_size=192, stride=128)

    for win in windows:
        seq = assemble_joint_sequence(
            doc_window_ids=win,
            code_ids=code_ids,
            tokenizer=tokenizer,
            max_length=512,
            code_truncation=truncation,
        )

        # Invariant 1: Strictly <= 512
        assert len(seq) <= 512, f"Sequence length {len(seq)} exceeded 512!"

        # Invariant 2: Starts with [CLS]
        assert seq[0] == tokenizer.cls_token_id

        # Invariant 3: Separators structure
        # [CLS] doc [SEP, SEP] code [SEP]
        sep = tokenizer.sep_token_id
        win_len = len(win)
        assert seq[win_len + 1] == sep
        assert seq[win_len + 2] == sep
        assert seq[-1] == sep


def test_window0_exact_parity_with_baseline():
    """Window 0 sequence must be bit-for-bit identical to baseline make_collate_fn sequence."""
    tokenizer = DummyTokenizer()
    doc_text = "word " * 300  # 300 tokens
    code_text = "code " * 600  # 600 tokens

    # Baseline collate_fn
    collate_fn = make_collate_fn(
        tokenizer=tokenizer,
        max_length=512,
        doc_max_tokens=192,
        truncation_strategy="head_tail",
    )
    batch = [(doc_text, code_text, 1, {})]
    inputs_baseline, _, _ = collate_fn(batch)
    baseline_seq = inputs_baseline["input_ids"][0].tolist()

    # Sliding window function
    windows = build_function_sliding_windows(
        docstring=doc_text,
        code=code_text,
        tokenizer=tokenizer,
        max_length=512,
        window_size=192,
        stride=128,
        code_truncation="head_tail",
    )
    window0_seq = windows[0]

    assert window0_seq == baseline_seq, "Window 0 sequence differs from baseline make_collate_fn!"
