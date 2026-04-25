"""
§3.3.4  Multi-granularity attribute-guided hierarchical prediction network.

Architecture
------------
The model is split into two attribute-recognition layers mirroring the
multi-granularity hierarchy discovered in §3.3.2:

  High-level layer (coarse granularity)
  ──────────────────────────────────────
  Predicts *coarse-grained* attributes (those identified as high-level by
  transfer-entropy analysis).  These are the more easily recognisable
  attributes that provide broad structural information about the flow state.

  Low-level layer  (fine granularity)
  ─────────────────────────────────────
  Predicts *fine-grained* attributes with the guidance of high-level
  attribute predictions.  An attribute feature guidance module (AFGM)
  injects the high-level predictions as additional context, enabling
  knowledge transfer from easy attributes to hard ones.

  Multi-level fusion inference
  ────────────────────────────
  Flow-state identity is inferred by comparing the predicted full attribute
  vector against all class prototype attribute vectors (§3.3.2) using cosine
  similarity.  Both coarse and fine predictions contribute to the final
  fused attribute vector, enabling more stable recognition.

Zero-shot capability
────────────────────
Because recognition is mediated by *attributes* rather than class labels,
the model can identify flow states whose class labels were never seen during
training, provided their attribute prototypes are known.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Attribute Feature Guidance Module (AFGM)
# ──────────────────────────────────────────────────────────────────────────────

class AttributeFeatureGuidanceModule(nn.Module):
    """
    Injects high-level attribute predictions as guidance for the low-level
    attribute recognition layer.

    Concretely, the high-level predictions are transformed into a guidance
    vector via a small MLP and added to the low-level feature representation
    before the low-level prediction head.

    Parameters
    ----------
    n_coarse   : number of coarse-grained (high-level) attributes.
    feature_dim: dimensionality of the fused feature vector.
    """

    def __init__(self, n_coarse: int, feature_dim: int) -> None:
        super().__init__()
        self.guidance_mlp = nn.Sequential(
            nn.Linear(n_coarse, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: torch.Tensor,       # (B, feature_dim)
        coarse_pred: torch.Tensor,    # (B, n_coarse)
    ) -> torch.Tensor:
        guidance = self.guidance_mlp(coarse_pred)  # (B, feature_dim)
        return features * guidance                 # element-wise gating


# ──────────────────────────────────────────────────────────────────────────────
# High-level (coarse) attribute recognition layer
# ──────────────────────────────────────────────────────────────────────────────

class HighLevelAttributeLayer(nn.Module):
    """
    Predicts coarse-grained attributes from the fused embedding.

    Parameters
    ----------
    feature_dim : input feature dimension.
    n_coarse    : number of coarse-grained attributes to predict.
    """

    def __init__(self, feature_dim: int, n_coarse: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, n_coarse),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # (B, n_coarse)


# ──────────────────────────────────────────────────────────────────────────────
# Low-level (fine) attribute recognition layer
# ──────────────────────────────────────────────────────────────────────────────

class LowLevelAttributeLayer(nn.Module):
    """
    Predicts fine-grained attributes guided by coarse predictions.

    Parameters
    ----------
    feature_dim : input feature dimension.
    n_fine      : number of fine-grained attributes to predict.
    n_coarse    : number of coarse attributes (for AFGM).
    """

    def __init__(self, feature_dim: int, n_fine: int, n_coarse: int) -> None:
        super().__init__()
        self.afgm = AttributeFeatureGuidanceModule(n_coarse, feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, n_fine),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: torch.Tensor,      # (B, feature_dim)
        coarse_pred: torch.Tensor,   # (B, n_coarse)
    ) -> torch.Tensor:
        guided = self.afgm(features, coarse_pred)
        return self.head(guided)  # (B, n_fine)


# ──────────────────────────────────────────────────────────────────────────────
# Hierarchical prediction network
# ──────────────────────────────────────────────────────────────────────────────

class HierarchicalAttributeNetwork(nn.Module):
    """
    §3.3.4  Multi-granularity attribute-guided hierarchical prediction network.

    Parameters
    ----------
    feature_dim    : dimensionality of the input fused feature vector.
    coarse_indices : list of attribute indices belonging to the coarse layer.
    fine_indices   : list of attribute indices belonging to the fine layer.
    """

    def __init__(
        self,
        feature_dim: int,
        coarse_indices: List[int],
        fine_indices: List[int],
    ) -> None:
        super().__init__()
        self.coarse_indices = coarse_indices
        self.fine_indices   = fine_indices
        n_coarse = len(coarse_indices)
        n_fine   = len(fine_indices)

        self.high_layer = HighLevelAttributeLayer(feature_dim, n_coarse)
        self.low_layer  = LowLevelAttributeLayer(feature_dim, n_fine, n_coarse)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        features : (B, feature_dim) fused embedding from FlowFeatureExtractor.

        Returns
        -------
        dict with keys:
            ``"coarse"``     – coarse attribute predictions (B, n_coarse)
            ``"fine"``       – fine attribute predictions   (B, n_fine)
            ``"attributes"`` – full attribute vector assembled from coarse+fine
                               predictions, ordered by original attribute index,
                               shape (B, n_coarse + n_fine).
        """
        coarse_pred = self.high_layer(features)
        fine_pred   = self.low_layer(features, coarse_pred)

        # Reconstruct full attribute vector in original attribute order
        n_total = len(self.coarse_indices) + len(self.fine_indices)
        B = features.shape[0]
        full_attr = torch.zeros(B, n_total, device=features.device)
        for i, idx in enumerate(self.coarse_indices):
            full_attr[:, idx] = coarse_pred[:, i]
        for i, idx in enumerate(self.fine_indices):
            full_attr[:, idx] = fine_pred[:, i]

        return {
            "coarse": coarse_pred,
            "fine":   fine_pred,
            "attributes": full_attr,
        }


# ──────────────────────────────────────────────────────────────────────────────
# ZSL classifier — multi-level fusion inference
# ──────────────────────────────────────────────────────────────────────────────

class ZSLFlowClassifier:
    """
    Zero-shot flow-state classifier based on attribute prototype matching.

    Flow-state identity is inferred by computing the cosine similarity
    between the predicted attribute vector and every class prototype
    attribute vector.  The class with the highest similarity is selected.

    Parameters
    ----------
    attribute_matrix : np.ndarray, shape (n_classes, n_attributes)
        Each row is the prototype attribute vector for one flow regime.
    class_names      : list of flow-regime name strings.
    fusion_weights   : tuple (w_coarse, w_fine)
        Relative weights when fusing coarse and fine attribute predictions.
        Defaults to (0.4, 0.6), giving slightly more weight to fine-grained.
    """

    def __init__(
        self,
        attribute_matrix: np.ndarray,
        class_names: List[str],
        fusion_weights: Tuple[float, float] = (0.4, 0.6),
    ) -> None:
        self.attribute_matrix = attribute_matrix.copy()
        self.class_names      = class_names
        self.fusion_weights   = fusion_weights

    # ------------------------------------------------------------------
    def predict(
        self,
        attr_pred: np.ndarray,
        coarse_pred: Optional[np.ndarray] = None,
        fine_pred: Optional[np.ndarray] = None,
        coarse_indices: Optional[List[int]] = None,
        fine_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Predict flow-regime class indices from predicted attribute vectors.

        Parameters
        ----------
        attr_pred      : (N, n_attributes) full attribute prediction.
        coarse_pred    : (N, n_coarse) optional coarse-layer output.
        fine_pred      : (N, n_fine)   optional fine-layer output.
        coarse_indices : attribute indices for coarse attributes.
        fine_indices   : attribute indices for fine attributes.

        Returns
        -------
        class_indices : np.ndarray, shape (N,)  integer class predictions.
        """
        if (
            coarse_pred is not None and fine_pred is not None
            and coarse_indices is not None and fine_indices is not None
        ):
            # Multi-level fusion: blend coarse and fine predictions
            fused = attr_pred.copy()
            w_c, w_f = self.fusion_weights
            for i, idx in enumerate(coarse_indices):
                fused[:, idx] = w_c * coarse_pred[:, i] + (1 - w_c) * attr_pred[:, idx]
            for i, idx in enumerate(fine_indices):
                fused[:, idx] = w_f * fine_pred[:, i] + (1 - w_f) * attr_pred[:, idx]
            query = fused
        else:
            query = attr_pred

        # Cosine similarity against class prototypes
        proto = self.attribute_matrix  # (n_classes, n_attributes)
        proto_norm = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-12)
        query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
        sim = query_norm @ proto_norm.T  # (N, n_classes)
        return np.argmax(sim, axis=1)

    # ------------------------------------------------------------------
    def predict_names(
        self,
        attr_pred: np.ndarray,
        **kwargs,
    ) -> List[str]:
        """Return class name strings instead of integer indices."""
        indices = self.predict(attr_pred, **kwargs)
        return [self.class_names[i] for i in indices]
