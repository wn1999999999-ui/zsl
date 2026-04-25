"""
§3.3.3  Manifold-regularised global-local spatiotemporal feature extractor.

Architecture
------------
The network combines two information streams:

  (a) Global stream  – CVA canonical variates encode long-range sequential
      correlations via a Temporal Convolutional / GRU encoder.

  (b) Local stream   – LPP-projected features encode local manifold geometry
      via a small fully-connected encoder.

Both streams are concatenated and passed through an attribute-supervised
projection head that maps process data to the attribute embedding space
(dim = n_attributes).  Attribute supervision drives the model to extract
features that are semantically aligned with the expert-defined attributes,
enabling zero-shot transfer to unseen flow regimes.

Complexity note
---------------
The network is intentionally lightweight (total parameters ≈ tens of thousands)
so that it trains quickly even without a GPU, while still being expressive
enough to capture non-linear structure.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class GlobalEncoder(nn.Module):
    """
    Temporal GRU encoder for global sequential correlations.

    Input  : (batch, seq_len, cva_dim)
    Output : (batch, hidden_dim)
    """

    def __init__(self, cva_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=cva_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, cva_dim)
        _, h = self.gru(x)   # h : (num_layers, B, hidden_dim)
        out = self.norm(h[-1])  # last layer hidden state
        return out  # (B, hidden_dim)


class LocalEncoder(nn.Module):
    """
    FC encoder for LPP-projected local manifold features.

    Input  : (batch, lpp_dim)
    Output : (batch, hidden_dim)
    """

    def __init__(self, lpp_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lpp_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, hidden_dim)


class AttributeProjectionHead(nn.Module):
    """
    Maps fused (global + local) features to the attribute embedding space.

    Input  : (batch, 2 * hidden_dim)
    Output : (batch, n_attributes)  – predicted attribute scores in [0, 1]
    """

    def __init__(self, in_dim: int, n_attributes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_dim // 2, n_attributes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, n_attributes)


# ──────────────────────────────────────────────────────────────────────────────
# Main feature extractor
# ──────────────────────────────────────────────────────────────────────────────

class FlowFeatureExtractor(nn.Module):
    """
    Manifold-regularised global-local spatiotemporal feature extractor (§3.3.3).

    Parameters
    ----------
    cva_dim      : dimensionality of CVA latent state (input to global encoder).
    lpp_dim      : dimensionality of LPP projection (input to local encoder).
    hidden_dim   : internal representation width for each stream.
    n_attributes : number of expert attributes to predict.
    """

    def __init__(
        self,
        cva_dim: int,
        lpp_dim: int,
        hidden_dim: int = 64,
        n_attributes: int = 12,
    ) -> None:
        super().__init__()
        self.global_enc = GlobalEncoder(cva_dim, hidden_dim)
        self.local_enc  = LocalEncoder(lpp_dim, hidden_dim)
        self.proj_head  = AttributeProjectionHead(2 * hidden_dim, n_attributes)

    def forward(
        self,
        cva_seq: torch.Tensor,   # (B, T, cva_dim)
        lpp_feat: torch.Tensor,  # (B, lpp_dim)
    ) -> dict:
        """
        Parameters
        ----------
        cva_seq  : global sequential features from CVA (time sequence).
        lpp_feat : local manifold features from LPP (per-sample vector).

        Returns
        -------
        dict with keys:
            ``"embedding"``  – fused feature vector (B, 2*hidden_dim)
            ``"attributes"`` – predicted attribute scores (B, n_attributes)
        """
        g = self.global_enc(cva_seq)   # (B, hidden_dim)
        l = self.local_enc(lpp_feat)   # (B, hidden_dim)
        fused = torch.cat([g, l], dim=-1)  # (B, 2*hidden_dim)
        attr_pred = self.proj_head(fused)  # (B, n_attributes)
        return {"embedding": fused, "attributes": attr_pred}


# ──────────────────────────────────────────────────────────────────────────────
# Trainer helper
# ──────────────────────────────────────────────────────────────────────────────

def train_feature_extractor(
    model: FlowFeatureExtractor,
    cva_seq_train: np.ndarray,   # (N, T, cva_dim)
    lpp_feat_train: np.ndarray,  # (N, lpp_dim)
    attr_labels_train: np.ndarray,  # (N, n_attributes)  values in {0, 0.5, 1}
    n_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> list:
    """
    Train the feature extractor with attribute-supervised BCE loss.

    Returns
    -------
    loss_history : list of float  (average loss per epoch)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    cva_t  = torch.tensor(cva_seq_train,    dtype=torch.float32)
    lpp_t  = torch.tensor(lpp_feat_train,   dtype=torch.float32)
    attr_t = torch.tensor(attr_labels_train, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(cva_t, lpp_t, attr_t)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    loss_history = []
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for cva_b, lpp_b, attr_b in loader:
            cva_b  = cva_b.to(device)
            lpp_b  = lpp_b.to(device)
            attr_b = attr_b.to(device)

            optimizer.zero_grad()
            out   = model(cva_b, lpp_b)
            loss  = criterion(out["attributes"], attr_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(cva_b)

        avg_loss = epoch_loss / len(dataset)
        loss_history.append(avg_loss)

    model.eval()
    return loss_history
