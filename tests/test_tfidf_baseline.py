#!/usr/bin/env python3
"""
tests/test_tfidf_baseline.py — Unit and Invariant Tests for TF-IDF Baseline.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.training.run_tfidf_baseline import (
    clean_docstring,
    compute_metrics,
    compute_relational_features,
    compute_score_distributions,
    extract_identifiers_from_code,
    extract_words_from_text,
    load_jsonl_dataset,
    sweep_validation_threshold,
)
from semdrift.data.labels import extract_label_tuple


def test_clean_docstring():
    raw = "[SUMMARY]\nDoes not validate the input credentials.\n[PARAMETERS]\nuser: username string\n[RETURNS]\nbool"
    cleaned = clean_docstring(raw)
    assert "Does not validate the input credentials." in cleaned
    assert "[SUMMARY]" not in cleaned
    assert "[PARAMETERS]" not in cleaned
    assert "not" in cleaned
    assert "user: username string" in cleaned


def test_extract_identifiers_from_code():
    code = """
def compute_hash(data_buffer, salt=None):
    result = hashlib.sha256(data_buffer).hexdigest()
    return result
"""
    identifiers = extract_identifiers_from_code(code)
    assert "compute_hash" in identifiers
    assert "data_buffer" in identifiers
    assert "salt" in identifiers
    assert "result" in identifiers


def test_extract_words_from_text():
    words = extract_words_from_text("Does not return None, raises ValueError!")
    assert "does" in words
    assert "not" in words
    assert "return" in words
    assert "none" in words
    assert "raises" in words
    assert "valueerror" in words


def test_compute_relational_features():
    docs = ["Validate user credentials and password", "Does not handle timeout error"]
    codes = [
        "def validate(user, password):\n    return True",
        "def run_query():\n    pass",
    ]
    feats = compute_relational_features(docs, codes)
    assert feats.shape == (2, 6)
    # Jaccard for first pair should be > 0 because 'validate', 'user', 'password' overlap
    assert feats[0, 0] > 0.0
    # Jaccard for second pair should be near 0 (no overlap)
    assert feats[1, 0] == 0.0


def test_stop_words_none_preservation():
    texts = [
        "does validate input",
        "does not validate input",
        "returns None",
        "raises ValueError",
    ]
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=None)
    X = vec.fit_transform(texts)
    vocab = vec.vocabulary_
    # Critical negation & operator bigrams MUST be preserved in vocabulary
    assert "does not" in vocab
    assert "not validate" in vocab
    assert "returns none" in vocab
    assert "raises valueerror" in vocab


def test_sweep_validation_threshold():
    y_true = [0, 0, 1, 1]
    # Probabilities where tau=0.60 gives perfect separation
    y_prob = np.array([0.10, 0.40, 0.70, 0.90])
    best_tau, best_f1, history = sweep_validation_threshold(y_true, y_prob)
    assert 0.40 < best_tau <= 0.70
    assert best_f1 == 1.0
    assert 0.50 in history


def test_compute_score_distributions():
    probs = [0.1, 0.2, 0.8, 0.9]
    groups = ["clean", "clean", "drift", "drift"]
    stats = compute_score_distributions(probs, groups)
    assert "clean" in stats
    assert "drift" in stats
    assert stats["clean"]["mean"] == pytest.approx(0.15, abs=1e-4)
    assert stats["drift"]["mean"] == pytest.approx(0.85, abs=1e-4)
    assert stats["clean"]["count"] == 2
    assert stats["drift"]["count"] == 2


def test_compute_metrics_structure():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    y_prob = [0.2, 0.6, 0.4, 0.8]
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics["count"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["pred_aligned"] == 2
    assert metrics["pred_drifted"] == 2
    assert metrics["roc_auc"] is not None


def test_load_jsonl_dataset_schema_invariants():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        # Invalid sample with conflicting labels
        f.write(json.dumps({"code": "def f(): pass", "docstring": "doc", "pseudo_label": 1, "label": "aligned"}) + "\n")
        bad_path = Path(f.name)

    try:
        with pytest.raises(ValueError):
            load_jsonl_dataset(bad_path)
    finally:
        bad_path.unlink()
