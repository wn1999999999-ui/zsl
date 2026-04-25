"""features sub-package — §3.3.3."""
from .cva_model import CVAStateSpaceModel
from .manifold_regularization import LocalPreservingProjection
from .feature_extractor import (
    FlowFeatureExtractor,
    GlobalEncoder,
    LocalEncoder,
    AttributeProjectionHead,
    train_feature_extractor,
)

__all__ = [
    "CVAStateSpaceModel",
    "LocalPreservingProjection",
    "FlowFeatureExtractor",
    "GlobalEncoder",
    "LocalEncoder",
    "AttributeProjectionHead",
    "train_feature_extractor",
]
