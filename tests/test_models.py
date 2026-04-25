"""Tests for §3.3.4 hierarchical network and ZSL classifier."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
from src.models import (
    HierarchicalAttributeNetwork,
    ZSLFlowClassifier,
    ZSLPipeline,
    train_zsl_pipeline,
    build_zsl_classifier,
)
from src.attributes import ATTRIBUTE_MATRIX, FLOW_REGIME_NAMES, NUM_ATTRIBUTES


COARSE_IDX = list(range(NUM_ATTRIBUTES // 2))
FINE_IDX   = list(range(NUM_ATTRIBUTES // 2, NUM_ATTRIBUTES))
FEATURE_DIM = 128  # 2 * hidden_dim with hidden_dim=64


def _rand_features(B=16, seed=5):
    rng = np.random.RandomState(seed)
    return torch.from_numpy(rng.randn(B, FEATURE_DIM).astype(np.float32))


class TestHierarchicalAttributeNetwork:
    def test_output_shapes(self):
        net = HierarchicalAttributeNetwork(
            feature_dim=FEATURE_DIM,
            coarse_indices=COARSE_IDX,
            fine_indices=FINE_IDX,
        )
        net.eval()
        features = _rand_features()
        with torch.no_grad():
            out = net(features)
        assert out["coarse"].shape     == (16, len(COARSE_IDX))
        assert out["fine"].shape       == (16, len(FINE_IDX))
        assert out["attributes"].shape == (16, NUM_ATTRIBUTES)

    def test_output_in_0_1(self):
        net = HierarchicalAttributeNetwork(
            feature_dim=FEATURE_DIM,
            coarse_indices=COARSE_IDX,
            fine_indices=FINE_IDX,
        )
        net.eval()
        with torch.no_grad():
            out = net(_rand_features())
        for key in ("coarse", "fine", "attributes"):
            arr = out[key].numpy()
            assert (arr >= 0.0).all() and (arr <= 1.0).all(), \
                f"Values out of [0,1] for key '{key}'"


class TestZSLFlowClassifier:
    def test_predict_seen_classes(self):
        clf = build_zsl_classifier()
        # Use perfect attribute prototypes as query → should predict correct class
        preds = clf.predict(ATTRIBUTE_MATRIX)
        expected = np.arange(len(FLOW_REGIME_NAMES))
        np.testing.assert_array_equal(preds, expected)

    def test_predict_returns_valid_indices(self):
        clf = build_zsl_classifier()
        rng = np.random.RandomState(0)
        query = rng.rand(10, NUM_ATTRIBUTES).astype(np.float32)
        preds = clf.predict(query)
        assert preds.shape == (10,)
        assert ((preds >= 0) & (preds < len(FLOW_REGIME_NAMES))).all()

    def test_predict_names_length(self):
        clf = build_zsl_classifier()
        rng = np.random.RandomState(1)
        query = rng.rand(5, NUM_ATTRIBUTES).astype(np.float32)
        names = clf.predict_names(query)
        assert len(names) == 5
        for name in names:
            assert name in FLOW_REGIME_NAMES

    def test_multi_level_fusion(self):
        clf = build_zsl_classifier()
        query = ATTRIBUTE_MATRIX.copy()
        coarse_pred = ATTRIBUTE_MATRIX[:, COARSE_IDX]
        fine_pred   = ATTRIBUTE_MATRIX[:, FINE_IDX]
        preds = clf.predict(
            query,
            coarse_pred=coarse_pred,
            fine_pred=fine_pred,
            coarse_indices=COARSE_IDX,
            fine_indices=FINE_IDX,
        )
        np.testing.assert_array_equal(preds, np.arange(len(FLOW_REGIME_NAMES)))


class TestZSLPipeline:
    def _make_data(self, N=64, T=8, cva_dim=8, lpp_dim=6, seed=42):
        rng = np.random.RandomState(seed)
        cva_seq  = rng.randn(N, T, cva_dim).astype(np.float32)
        lpp_feat = rng.randn(N, lpp_dim).astype(np.float32)
        y_idx    = rng.randint(0, len(FLOW_REGIME_NAMES), size=N)
        attr_lbl = ATTRIBUTE_MATRIX[y_idx].astype(np.float32)
        return cva_seq, lpp_feat, attr_lbl

    def test_forward_shapes(self):
        cva_seq, lpp_feat, _ = self._make_data()
        pipeline = ZSLPipeline(
            cva_dim=8, lpp_dim=6, hidden_dim=32,
            coarse_indices=COARSE_IDX, fine_indices=FINE_IDX,
        )
        pipeline.eval()
        with torch.no_grad():
            out = pipeline(
                torch.from_numpy(cva_seq),
                torch.from_numpy(lpp_feat),
            )
        assert out["embedding"].shape        == (64, 64)
        assert out["attributes"].shape       == (64, NUM_ATTRIBUTES)
        assert out["coarse"].shape           == (64, len(COARSE_IDX))
        assert out["fine"].shape             == (64, len(FINE_IDX))
        assert out["hier_attributes"].shape  == (64, NUM_ATTRIBUTES)

    def test_training_reduces_loss(self):
        cva_seq, lpp_feat, attr_lbl = self._make_data(N=128)
        pipeline = ZSLPipeline(
            cva_dim=8, lpp_dim=6, hidden_dim=32,
            coarse_indices=COARSE_IDX, fine_indices=FINE_IDX,
        )
        losses = train_zsl_pipeline(
            pipeline, cva_seq, lpp_feat, attr_lbl,
            n_epochs=5, batch_size=32,
        )
        assert losses[-1] < losses[0], "Loss should decrease during training."
