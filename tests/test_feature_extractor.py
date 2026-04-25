"""Tests for §3.3.3 deep-learning feature extractor."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
from src.features.feature_extractor import (
    FlowFeatureExtractor,
    train_feature_extractor,
)
from src.attributes import ATTRIBUTE_MATRIX, FLOW_REGIME_NAMES, NUM_ATTRIBUTES


def _make_batch(B=8, T=10, cva_dim=8, lpp_dim=6, n_classes=6, seed=0):
    rng = np.random.RandomState(seed)
    cva_seq  = rng.randn(B, T, cva_dim).astype(np.float32)
    lpp_feat = rng.randn(B, lpp_dim).astype(np.float32)
    y        = rng.randint(0, n_classes, size=B)
    attr     = ATTRIBUTE_MATRIX[y].astype(np.float32)
    return cva_seq, lpp_feat, attr


class TestFlowFeatureExtractor:
    def test_forward_shapes(self):
        cva_seq, lpp_feat, attr = _make_batch()
        model = FlowFeatureExtractor(cva_dim=8, lpp_dim=6, hidden_dim=32)
        model.eval()
        with torch.no_grad():
            out = model(
                torch.from_numpy(cva_seq),
                torch.from_numpy(lpp_feat),
            )
        assert out["embedding"].shape  == (8, 64)
        assert out["attributes"].shape == (8, NUM_ATTRIBUTES)

    def test_attributes_in_0_1(self):
        cva_seq, lpp_feat, _ = _make_batch()
        model = FlowFeatureExtractor(cva_dim=8, lpp_dim=6, hidden_dim=32)
        model.eval()
        with torch.no_grad():
            out = model(
                torch.from_numpy(cva_seq),
                torch.from_numpy(lpp_feat),
            )
        attrs = out["attributes"].numpy()
        assert (attrs >= 0.0).all() and (attrs <= 1.0).all()

    def test_training_reduces_loss(self):
        rng = np.random.RandomState(42)
        N, T, cva_dim, lpp_dim = 128, 10, 8, 6
        cva_seq  = rng.randn(N, T, cva_dim).astype(np.float32)
        lpp_feat = rng.randn(N, lpp_dim).astype(np.float32)
        y_idx    = rng.randint(0, 6, size=N)
        attr_lbl = ATTRIBUTE_MATRIX[y_idx].astype(np.float32)

        model  = FlowFeatureExtractor(cva_dim=cva_dim, lpp_dim=lpp_dim, hidden_dim=32)
        losses = train_feature_extractor(
            model, cva_seq, lpp_feat, attr_lbl, n_epochs=5, batch_size=32
        )
        assert losses[-1] < losses[0], "Loss should decrease during training."
