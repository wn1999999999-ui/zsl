"""Tests for §3.3.2 transfer entropy and granularity inference."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.attributes import (
    ATTRIBUTE_NAMES,
    NUM_ATTRIBUTES,
    transfer_entropy,
    compute_te_matrix,
    infer_granularity_layers,
    TransferEntropyAnalyzer,
)


class TestTransferEntropy:
    def test_te_non_negative(self):
        rng = np.random.RandomState(42)
        x = rng.randn(500)
        y = rng.randn(500)
        te = transfer_entropy(x, y)
        assert te >= 0.0

    def test_te_causal_direction(self):
        """TE(X->Y) should be larger when X causally drives Y."""
        rng = np.random.RandomState(1)
        x = rng.randn(500)
        # y is causally driven by x (autoregressive)
        y = np.zeros(500)
        y[0] = rng.randn()
        for t in range(1, 500):
            y[t] = 0.8 * x[t - 1] + 0.1 * rng.randn()
        te_xy = transfer_entropy(x, y)
        te_yx = transfer_entropy(y, x)
        assert te_xy > te_yx, (
            f"Expected TE(X->Y)={te_xy:.4f} > TE(Y->X)={te_yx:.4f}"
        )

    def test_te_short_series_returns_zero(self):
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        assert transfer_entropy(x, y) == 0.0

    def test_te_with_nan(self):
        rng = np.random.RandomState(7)
        x = rng.randn(100)
        y = rng.randn(100)
        x[10] = np.nan
        y[20] = np.nan
        te = transfer_entropy(x, y)
        assert np.isfinite(te)


class TestTEMatrix:
    def test_te_matrix_shape(self):
        rng = np.random.RandomState(0)
        scores = rng.randn(200, NUM_ATTRIBUTES).astype(np.float32)
        te_mat = compute_te_matrix(scores)
        assert te_mat.shape == (NUM_ATTRIBUTES, NUM_ATTRIBUTES)

    def test_te_matrix_diagonal_zero(self):
        rng = np.random.RandomState(0)
        scores = rng.randn(200, NUM_ATTRIBUTES).astype(np.float32)
        te_mat = compute_te_matrix(scores)
        np.testing.assert_array_equal(np.diag(te_mat), np.zeros(NUM_ATTRIBUTES))

    def test_te_matrix_non_negative(self):
        rng = np.random.RandomState(0)
        scores = rng.randn(200, NUM_ATTRIBUTES).astype(np.float32)
        te_mat = compute_te_matrix(scores)
        assert (te_mat >= 0).all()


class TestGranularityInference:
    def test_granularity_partitions_all_attributes(self):
        rng = np.random.RandomState(0)
        scores = rng.randn(200, NUM_ATTRIBUTES).astype(np.float32)
        te_mat = compute_te_matrix(scores)
        layers = infer_granularity_layers(te_mat, ATTRIBUTE_NAMES)
        all_attrs = set(layers["coarse"]) | set(layers["fine"])
        assert all_attrs == set(ATTRIBUTE_NAMES)

    def test_granularity_no_overlap(self):
        rng = np.random.RandomState(0)
        scores = rng.randn(200, NUM_ATTRIBUTES).astype(np.float32)
        te_mat = compute_te_matrix(scores)
        layers = infer_granularity_layers(te_mat, ATTRIBUTE_NAMES)
        overlap = set(layers["coarse"]) & set(layers["fine"])
        assert len(overlap) == 0


class TestTransferEntropyAnalyzer:
    def test_fit_and_access(self):
        rng = np.random.RandomState(3)
        scores = rng.randn(150, NUM_ATTRIBUTES).astype(np.float32)
        analyzer = TransferEntropyAnalyzer()
        analyzer.fit(scores, ATTRIBUTE_NAMES)
        assert analyzer.te_matrix.shape == (NUM_ATTRIBUTES, NUM_ATTRIBUTES)
        assert "coarse" in analyzer.granularity
        assert "fine" in analyzer.granularity

    def test_not_fitted_raises(self):
        analyzer = TransferEntropyAnalyzer()
        with pytest.raises(RuntimeError):
            _ = analyzer.te_matrix
        with pytest.raises(RuntimeError):
            _ = analyzer.granularity
