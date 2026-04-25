"""Tests for §3.3.2 attribute definition and LDA feature extraction."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.attributes import (
    ATTRIBUTE_NAMES,
    ATTRIBUTE_MATRIX,
    FLOW_REGIME_NAMES,
    NUM_ATTRIBUTES,
    NUM_CLASSES,
    get_attribute_matrix,
    get_attribute_vector,
    LDAAttributeExtractor,
)


class TestAttributeDefinition:
    def test_num_attributes_ge_10(self):
        assert NUM_ATTRIBUTES >= 10

    def test_num_classes(self):
        assert NUM_CLASSES == len(FLOW_REGIME_NAMES)

    def test_matrix_shape(self):
        A = get_attribute_matrix()
        assert A.shape == (NUM_CLASSES, NUM_ATTRIBUTES)

    def test_matrix_values_valid(self):
        A = get_attribute_matrix()
        valid = {0.0, 0.5, 1.0}
        for v in A.ravel():
            assert v in valid, f"Unexpected attribute value: {v}"

    def test_get_attribute_vector(self):
        for c in range(NUM_CLASSES):
            v = get_attribute_vector(c)
            assert v.shape == (NUM_ATTRIBUTES,)

    def test_attribute_names_length(self):
        assert len(ATTRIBUTE_NAMES) == NUM_ATTRIBUTES

    def test_matrix_is_copy(self):
        A1 = get_attribute_matrix()
        A2 = get_attribute_matrix()
        A1[0, 0] = -999
        assert A2[0, 0] != -999


class TestLDAAttributeExtractor:
    def _make_data(self, n=300, d=20, n_classes=6, n_attr=12, seed=0):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, d).astype(np.float32)
        # Assign attribute labels: class c gets attribute vector from matrix
        y = rng.randint(0, n_classes, size=n)
        A = get_attribute_matrix()
        attr_labels = A[y]  # (n, n_attr)
        return X, attr_labels

    def test_fit_transform_shape(self):
        X, attr_labels = self._make_data()
        lda = LDAAttributeExtractor()
        scores = lda.fit_transform(X, attr_labels)
        assert scores.shape == (X.shape[0], NUM_ATTRIBUTES)

    def test_transform_matches_fit_transform(self):
        X, attr_labels = self._make_data()
        lda = LDAAttributeExtractor()
        s1 = lda.fit_transform(X, attr_labels)
        s2 = lda.transform(X)
        np.testing.assert_allclose(s1, s2, rtol=1e-5)

    def test_wrong_n_samples_raises(self):
        X, attr_labels = self._make_data(n=100)
        lda = LDAAttributeExtractor()
        with pytest.raises(ValueError):
            lda.fit(X[:50], attr_labels)

    def test_wrong_n_attributes_raises(self):
        X, attr_labels = self._make_data()
        lda = LDAAttributeExtractor()
        with pytest.raises(ValueError):
            lda.fit(X, attr_labels[:, :5])

    def test_transform_before_fit_raises(self):
        lda = LDAAttributeExtractor()
        with pytest.raises(Exception):
            lda.transform(np.zeros((10, 20), dtype=np.float32))
