"""
§3.3.2  Transfer entropy and multi-granularity attribute hierarchy.

Transfer entropy (TE) from time-series X → Y measures the amount of
information that the past of X provides about the future of Y beyond
what the past of Y already provides.

    TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

where *H* denotes Shannon entropy.

We estimate TE using binned histograms (symbolic / ordinal discretisation).
From the asymmetric TE matrix we derive a directed causal graph among
attributes and then partition attributes into coarse (high-level) and
fine (low-level) granularity layers.

Granularity rule:
    - Attributes whose *average outgoing* TE (net causal influence on others)
      exceeds the median across all attributes are placed in the **coarse**
      (high-level) layer.
    - The remaining attributes form the **fine** (low-level) layer.
"""

import numpy as np
from typing import Dict, List, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Low-level entropy helpers
# ──────────────────────────────────────────────────────────────────────────────

def _symbolise(x: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Bin a 1-D array into integer symbols in [0, n_bins-1]."""
    edges = np.linspace(np.nanmin(x), np.nanmax(x) + 1e-12, n_bins + 1)
    return np.digitize(x, edges[1:-1])


def _joint_prob(a: np.ndarray, b: np.ndarray, n_bins: int) -> np.ndarray:
    """Empirical joint probability table p(a, b)."""
    counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    for ai, bi in zip(a, b):
        counts[ai, bi] += 1
    counts /= counts.sum()
    return counts


def _cond_entropy(a: np.ndarray, b: np.ndarray, n_bins: int) -> float:
    """H(A | B)  from integer arrays a, b."""
    pab = _joint_prob(a, b, n_bins)
    pb = pab.sum(axis=0, keepdims=True) + 1e-15
    ratio = np.where(pab > 0, pab / pb, 1.0)
    return -np.sum(pab * np.log2(ratio + 1e-15))


def transfer_entropy(
    x: np.ndarray,
    y: np.ndarray,
    lag: int = 1,
    n_bins: int = 4,
) -> float:
    """
    Estimate TE(X → Y) using a first-order Markov approximation.

    Parameters
    ----------
    x, y : 1-D arrays of length T (the 1-D LDA scores for two attributes).
    lag  : time lag (default 1 sample).
    n_bins : number of histogram bins.

    Returns
    -------
    float  Transfer entropy TE(X→Y) in bits (≥ 0 by construction after
           clipping numerical noise).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Remove samples where either series is NaN
    valid = (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[valid], y[valid]

    T = len(y)
    if T <= lag + 1:
        return 0.0

    y_future = _symbolise(y[lag:], n_bins)       # Y_{t+lag}
    y_past   = _symbolise(y[:-lag], n_bins)       # Y_t
    x_past   = _symbolise(x[:-lag], n_bins)       # X_t

    # TE = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    #    = I(Y_future ; X_past | Y_past)
    h_y_given_ypast = _cond_entropy(y_future, y_past, n_bins)

    # Build 3-way joint: p(y_future, y_past, x_past)
    counts3 = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
    for yf, yp, xp in zip(y_future, y_past, x_past):
        counts3[yf, yp, xp] += 1
    counts3 /= counts3.sum()

    # H(Y_future | Y_past, X_past)
    pyx = counts3.sum(axis=0) + 1e-15  # p(y_past, x_past)
    ratio = np.where(counts3 > 0, counts3 / pyx[np.newaxis, :, :], 1.0)
    h_y_given_ypast_xpast = -np.sum(counts3 * np.log2(ratio + 1e-15))

    te = h_y_given_ypast - h_y_given_ypast_xpast
    return float(max(te, 0.0))


# ──────────────────────────────────────────────────────────────────────────────
# TE matrix and granularity inference
# ──────────────────────────────────────────────────────────────────────────────

def compute_te_matrix(
    scores: np.ndarray,
    lag: int = 1,
    n_bins: int = 4,
) -> np.ndarray:
    """
    Compute the pairwise transfer-entropy matrix from 1-D LDA scores.

    Parameters
    ----------
    scores : np.ndarray, shape (T, n_attributes)
        1-D LDA discriminant scores per attribute (rows = time steps).
    lag    : time lag used in TE estimation.
    n_bins : histogram bins for TE estimation.

    Returns
    -------
    te_matrix : np.ndarray, shape (n_attributes, n_attributes)
        te_matrix[i, j] = TE(attribute_i → attribute_j).
        Diagonal entries are 0.
    """
    n_attr = scores.shape[1]
    te_matrix = np.zeros((n_attr, n_attr), dtype=np.float32)
    for i in range(n_attr):
        for j in range(n_attr):
            if i != j:
                te_matrix[i, j] = transfer_entropy(
                    scores[:, i], scores[:, j], lag=lag, n_bins=n_bins
                )
    return te_matrix


def infer_granularity_layers(
    te_matrix: np.ndarray,
    attribute_names: List[str],
) -> Dict[str, List[str]]:
    """
    Partition attributes into coarse (high-level) and fine (low-level)
    granularity layers based on average outgoing transfer entropy.

    Parameters
    ----------
    te_matrix      : (n_attr, n_attr) transfer-entropy matrix.
    attribute_names: list of attribute name strings.

    Returns
    -------
    dict with keys ``"coarse"`` and ``"fine"``, each mapping to a list of
    attribute names.
    """
    avg_outgoing = te_matrix.sum(axis=1)  # sum of TE from attribute i to others
    threshold = np.median(avg_outgoing)

    coarse = [
        name for name, val in zip(attribute_names, avg_outgoing)
        if val >= threshold
    ]
    fine = [
        name for name, val in zip(attribute_names, avg_outgoing)
        if val < threshold
    ]
    return {"coarse": coarse, "fine": fine}


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

class TransferEntropyAnalyzer:
    """
    End-to-end wrapper: compute TE matrix and derive attribute granularity.

    Parameters
    ----------
    lag    : time lag for TE estimation (default 1).
    n_bins : histogram bins (default 4).
    """

    def __init__(self, lag: int = 1, n_bins: int = 4) -> None:
        self.lag = lag
        self.n_bins = n_bins
        self.te_matrix_: np.ndarray | None = None
        self.granularity_: Dict[str, List[str]] | None = None

    def fit(
        self,
        scores: np.ndarray,
        attribute_names: List[str],
    ) -> "TransferEntropyAnalyzer":
        """
        Fit transfer-entropy analysis on 1-D LDA scores.

        Parameters
        ----------
        scores          : (T, n_attributes) LDA score matrix.
        attribute_names : list of attribute name strings.
        """
        self.te_matrix_ = compute_te_matrix(scores, self.lag, self.n_bins)
        self.granularity_ = infer_granularity_layers(self.te_matrix_, attribute_names)
        return self

    @property
    def te_matrix(self) -> np.ndarray:
        if self.te_matrix_ is None:
            raise RuntimeError(
                "TransferEntropyAnalyzer must be fitted before accessing te_matrix. "
                "Call fit() first."
            )
        return self.te_matrix_

    @property
    def granularity(self) -> Dict[str, List[str]]:
        if self.granularity_ is None:
            raise RuntimeError(
                "TransferEntropyAnalyzer must be fitted before accessing granularity. "
                "Call fit() first."
            )
        return self.granularity_
