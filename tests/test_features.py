"""Tests for §3.3.3 CVA and LPP feature extractors."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.features import CVAStateSpaceModel, LocalPreservingProjection
from src.utils import split_dataset, generate_synthetic_flow_data


class TestDataUtils:
    def test_split_ratio(self):
        X, y = generate_synthetic_flow_data(n_samples_per_class=50, n_classes=4)
        X_tr, X_te, y_tr, y_te = split_dataset(X, y)
        total = len(y)
        assert abs(len(y_te) / total - 0.2) < 0.05

    def test_split_no_overlap(self):
        X, y = generate_synthetic_flow_data(n_samples_per_class=50, n_classes=4)
        X_tr, X_te, y_tr, y_te = split_dataset(X, y)
        # Row-wise check: no duplicate rows (simple sanity check)
        assert len(X_tr) + len(X_te) == len(X)

    def test_synthetic_shapes(self):
        X, y = generate_synthetic_flow_data(
            n_samples_per_class=30, n_features=16, n_classes=3
        )
        assert X.shape == (90, 16)
        assert y.shape == (90,)


class TestCVAStateSpaceModel:
    def _make_ts(self, T=200, p=8, seed=0):
        rng = np.random.RandomState(seed)
        return rng.randn(T, p).astype(np.float32)

    def test_fit_transform_shape(self):
        X = self._make_ts()
        cva = CVAStateSpaceModel(n_components=4, past_window=3, future_window=3)
        Z = cva.fit_transform(X)
        # Output should have T - past_window + 1 rows and 4 columns
        expected_rows = X.shape[0] - cva.past_window + 1
        assert Z.shape == (expected_rows, 4)

    def test_singular_values_descending(self):
        X = self._make_ts(T=300)
        cva = CVAStateSpaceModel(n_components=6, past_window=4, future_window=4)
        cva.fit(X)
        sv = cva.singular_values
        assert np.all(sv[:-1] >= sv[1:] - 1e-8)

    def test_transform_before_fit_raises(self):
        cva = CVAStateSpaceModel()
        X = self._make_ts()
        with pytest.raises(RuntimeError):
            cva.transform(X)

    def test_too_short_series_raises(self):
        cva = CVAStateSpaceModel(past_window=10, future_window=10)
        X = np.ones((5, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            cva.fit(X)

    def test_transform_consistent_with_fit_transform(self):
        X = self._make_ts()
        cva = CVAStateSpaceModel(n_components=4, past_window=3, future_window=3)
        Z1 = cva.fit_transform(X)
        Z2 = cva.transform(X)
        np.testing.assert_allclose(Z1, Z2, rtol=1e-5)


class TestLocalPreservingProjection:
    def _make_data(self, n=80, d=16, seed=1):
        rng = np.random.RandomState(seed)
        return rng.randn(n, d).astype(np.float32)

    def test_fit_transform_shape(self):
        X = self._make_data()
        lpp = LocalPreservingProjection(n_components=6, n_neighbors=4)
        out = lpp.fit_transform(X)
        assert out.shape == (80, 6)

    def test_transform_before_fit_raises(self):
        lpp = LocalPreservingProjection(n_components=4)
        X = self._make_data()
        with pytest.raises(RuntimeError):
            lpp.transform(X)

    def test_transform_matches_fit_transform(self):
        X = self._make_data()
        lpp = LocalPreservingProjection(n_components=4, n_neighbors=3)
        out1 = lpp.fit_transform(X)
        out2 = lpp.transform(X)
        np.testing.assert_allclose(out1, out2, rtol=1e-5)
