"""attributes sub-package — §3.3.2."""
from .attribute_definition import (
    ATTRIBUTE_NAMES,
    ATTRIBUTE_MATRIX,
    FLOW_REGIME_NAMES,
    NUM_ATTRIBUTES,
    NUM_CLASSES,
    get_attribute_matrix,
    get_attribute_vector,
)
from .lda_feature_extractor import LDAAttributeExtractor
from .transfer_entropy import (
    TransferEntropyAnalyzer,
    compute_te_matrix,
    infer_granularity_layers,
    transfer_entropy,
)

__all__ = [
    "ATTRIBUTE_NAMES",
    "ATTRIBUTE_MATRIX",
    "FLOW_REGIME_NAMES",
    "NUM_ATTRIBUTES",
    "NUM_CLASSES",
    "get_attribute_matrix",
    "get_attribute_vector",
    "LDAAttributeExtractor",
    "TransferEntropyAnalyzer",
    "compute_te_matrix",
    "infer_granularity_layers",
    "transfer_entropy",
]
