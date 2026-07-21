"""Tests for the semdrift.embedder module (Model A Baseline)."""

import pytest
import torch

from semdrift.embedder.embed import (
    compute_divergence,
    embed_function_record,
    get_embeddings,
    predict_drift,
    load_config,
)


class TestEmbedder:
    """Comprehensive tests for semdrift.embedder."""

    def test_load_config(self):
        """Ensure config is loaded properly or defaults are returned."""
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_compute_divergence_identical(self):
        """Identical embeddings must give divergence 0.0."""
        emb1 = torch.tensor([[1.0, 2.0, 3.0]])
        emb2 = torch.tensor([[1.0, 2.0, 3.0]])
        div = compute_divergence(emb1, emb2)
        assert abs(div - 0.0) < 1e-5

    def test_compute_divergence_orthogonal(self):
        """Orthogonal embeddings must give divergence 1.0."""
        emb1 = torch.tensor([[1.0, 0.0]])
        emb2 = torch.tensor([[0.0, 1.0]])
        div = compute_divergence(emb1, emb2)
        assert abs(div - 1.0) < 1e-5

    def test_compute_divergence_opposite(self):
        """Opposite embeddings must give divergence 2.0."""
        emb1 = torch.tensor([[1.0, 0.0]])
        emb2 = torch.tensor([[-1.0, 0.0]])
        div = compute_divergence(emb1, emb2)
        assert abs(div - 2.0) < 1e-5

    def test_predict_drift(self):
        """Test threshold-based drift classification."""
        assert predict_drift(0.20, threshold=0.15) == "drifted"
        assert predict_drift(0.10, threshold=0.15) == "aligned"

    def test_get_embeddings_shape(self):
        """Ensure get_embeddings returns shape (N, 768)."""
        texts = ["def foo(): pass", "Return sum of a and b."]
        embs = get_embeddings(texts, batch_size=2, pooling="mean")
        assert embs.shape == (2, 768)

    def test_get_embeddings_cls_pooling(self):
        """Ensure CLS pooling runs without error."""
        texts = ["def bar(): pass"]
        embs = get_embeddings(texts, batch_size=1, pooling="cls")
        assert embs.shape == (1, 768)

    def test_embed_function_record(self):
        """Ensure embed_function_record processes a dictionary record."""
        record = {
            "function_id": "test_func_001",
            "code": "def square(x):\n    return x * x",
            "docstring": "Compute the square of a number.",
        }
        c_emb, d_emb = embed_function_record(record, pooling="mean")
        assert c_emb.shape == (1, 768)
        assert d_emb.shape == (1, 768)
        div = compute_divergence(c_emb, d_emb)
        assert isinstance(div, float)

    def test_compute_divergence_mean_centering(self):
        """Ensure mean-centering shifts distributions prior to L2 normalization."""
        emb1 = torch.tensor([[10.0, 10.0], [12.0, 10.0], [8.0, 10.0]])
        emb2 = torch.tensor([[10.0, 10.0], [10.0, 12.0], [10.0, 8.0]])
        divs_raw = compute_divergence(emb1, emb2, normalize=True, mean_center=False)
        divs_centered = compute_divergence(emb1, emb2, normalize=True, mean_center=True)
        assert not torch.allclose(divs_raw, divs_centered)
        assert divs_centered.shape == (3,)
