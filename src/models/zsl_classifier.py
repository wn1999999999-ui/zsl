"""
§3.3.4  End-to-end ZSL trainer that wires together §3.3.3 and §3.3.4.

``ZSLPipeline`` provides a single high-level object that:

1. Prepares CVA + LPP inputs from raw sensor data.
2. Runs the FlowFeatureExtractor to obtain fused embeddings.
3. Passes embeddings through the HierarchicalAttributeNetwork to predict
   coarse and fine attributes.
4. Uses the ZSLFlowClassifier to recognise flow states (including unseen ones).

Training
--------
The overall loss combines:
    L_total = L_coarse + λ * L_fine
where both terms are binary cross-entropy against the attribute labels.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Dict

from ..attributes import (
    ATTRIBUTE_NAMES,
    ATTRIBUTE_MATRIX,
    FLOW_REGIME_NAMES,
    NUM_ATTRIBUTES,
)
from ..features.feature_extractor import FlowFeatureExtractor
from .attribute_network import HierarchicalAttributeNetwork, ZSLFlowClassifier


class ZSLPipeline(nn.Module):
    """
    Full ZSL pipeline: feature extraction + hierarchical attribute prediction.

    Parameters
    ----------
    cva_dim        : CVA latent dimensionality.
    lpp_dim        : LPP projection dimensionality.
    hidden_dim     : width of the feature encoder streams.
    coarse_indices : attribute indices for the coarse (high-level) layer.
    fine_indices   : attribute indices for the fine (low-level) layer.
    n_attributes   : total number of attributes.
    """

    def __init__(
        self,
        cva_dim: int,
        lpp_dim: int,
        hidden_dim: int = 64,
        coarse_indices: Optional[List[int]] = None,
        fine_indices: Optional[List[int]] = None,
        n_attributes: int = NUM_ATTRIBUTES,
    ) -> None:
        super().__init__()

        # Default: first half coarse, second half fine
        if coarse_indices is None:
            coarse_indices = list(range(n_attributes // 2))
        if fine_indices is None:
            fine_indices = list(range(n_attributes // 2, n_attributes))

        self.coarse_indices = coarse_indices
        self.fine_indices   = fine_indices

        feature_dim = 2 * hidden_dim

        self.feature_extractor = FlowFeatureExtractor(
            cva_dim=cva_dim,
            lpp_dim=lpp_dim,
            hidden_dim=hidden_dim,
            n_attributes=n_attributes,
        )
        self.hier_network = HierarchicalAttributeNetwork(
            feature_dim=feature_dim,
            coarse_indices=coarse_indices,
            fine_indices=fine_indices,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        cva_seq: torch.Tensor,   # (B, T, cva_dim)
        lpp_feat: torch.Tensor,  # (B, lpp_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            ``"embedding"``  – fused feature vector (B, 2*hidden_dim)
            ``"attributes"`` – full attribute prediction from extractor
            ``"coarse"``     – coarse attribute prediction (B, n_coarse)
            ``"fine"``       – fine attribute prediction (B, n_fine)
            ``"hier_attributes"`` – full attribute vector assembled from hier network
        """
        feat_out = self.feature_extractor(cva_seq, lpp_feat)
        hier_out = self.hier_network(feat_out["embedding"])
        return {**feat_out, **hier_out, "hier_attributes": hier_out["attributes"]}


def train_zsl_pipeline(
    pipeline: ZSLPipeline,
    cva_seq_train: np.ndarray,      # (N, T, cva_dim)
    lpp_feat_train: np.ndarray,     # (N, lpp_dim)
    attr_labels_train: np.ndarray,  # (N, n_attributes)
    n_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    fine_loss_weight: float = 1.5,
    device: Optional[str] = None,
) -> List[float]:
    """
    Train the full ZSL pipeline with hierarchical attribute supervision.

    Loss:
        L_total = L_extractor + L_coarse + fine_loss_weight * L_fine

    Returns
    -------
    loss_history : list of float (average loss per epoch)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = pipeline.to(device)
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCELoss()

    cva_t  = torch.tensor(cva_seq_train,    dtype=torch.float32)
    lpp_t  = torch.tensor(lpp_feat_train,   dtype=torch.float32)
    attr_t = torch.tensor(attr_labels_train, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(cva_t, lpp_t, attr_t)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    coarse_idx = pipeline.coarse_indices
    fine_idx   = pipeline.fine_indices

    loss_history = []
    pipeline.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for cva_b, lpp_b, attr_b in loader:
            cva_b  = cva_b.to(device)
            lpp_b  = lpp_b.to(device)
            attr_b = attr_b.to(device)

            optimizer.zero_grad()
            out = pipeline(cva_b, lpp_b)

            # Attribute extractor loss (full)
            l_ext    = bce(out["attributes"], attr_b)

            # Hierarchical losses
            l_coarse = bce(out["coarse"], attr_b[:, coarse_idx])
            l_fine   = bce(out["fine"],   attr_b[:, fine_idx])

            loss = l_ext + l_coarse + fine_loss_weight * l_fine
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(cva_b)

        avg_loss = epoch_loss / len(dataset)
        loss_history.append(avg_loss)

    pipeline.eval()
    return loss_history


def build_zsl_classifier(
    attribute_matrix: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> ZSLFlowClassifier:
    """
    Construct a ZSLFlowClassifier using the default attribute prototype matrix.
    """
    if attribute_matrix is None:
        attribute_matrix = ATTRIBUTE_MATRIX
    if class_names is None:
        class_names = FLOW_REGIME_NAMES
    return ZSLFlowClassifier(
        attribute_matrix=attribute_matrix,
        class_names=class_names,
    )
