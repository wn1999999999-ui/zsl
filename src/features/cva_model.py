"""
§3.3.3  Canonical Correlation Analysis (CVA) state-space model.

CVA maximises the correlation between *past* (Hankel) vectors and *future*
(Hankel) vectors of a multivariate time series, thereby revealing the global
sequential / dynamic structure of the process.

Algorithm
---------
1. Build past-window matrix  Y_past  and future-window matrix  Y_future
   from the raw multivariate signal using a sliding Hankel window.
2. Whiten both matrices.
3. Compute the SVD of the cross-covariance matrix to obtain canonical
   variates (latent state directions).
4. Project data onto the top-*k* canonical directions as the global
   latent state representation.

Reference: Larimore (1990), "Canonical variate analysis in identification,
filtering, and adaptive control."
"""

import numpy as np
from typing import Optional


def _hankel_matrix(X: np.ndarray, window: int) -> np.ndarray:
    """
    Build a Hankel (lag-window) matrix from a multivariate time series.

    Parameters
    ----------
    X      : (T, p) time series matrix.
    window : number of lags to stack.

    Returns
    -------
    H : (T - window + 1, p * window)
    """
    T, p = X.shape
    rows = T - window + 1
    H = np.empty((rows, p * window), dtype=X.dtype)
    for i in range(rows):
        H[i] = X[i: i + window].ravel()
    return H


def _whiten(M: np.ndarray, eps: float = 1e-9) -> tuple:
    """Return whitened matrix and the whitening transform."""
    cov = M.T @ M / M.shape[0]
    U, s, _ = np.linalg.svd(cov, full_matrices=False)
    s_inv_sqrt = 1.0 / np.sqrt(s + eps)
    W = U * s_inv_sqrt  # (d, d) whitening matrix (column directions)
    return M @ W, W


class CVAStateSpaceModel:
    """
    CVA-based global state-space model.

    Parameters
    ----------
    n_components : int
        Number of canonical variates / latent states to retain.
    past_window  : int
        Number of past time steps used in the Hankel window (default 5).
    future_window: int
        Number of future time steps used in the Hankel window (default 5).
    """

    def __init__(
        self,
        n_components: int = 8,
        past_window: int = 5,
        future_window: int = 5,
    ) -> None:
        self.n_components = n_components
        self.past_window = past_window
        self.future_window = future_window

        self._W_past: Optional[np.ndarray] = None
        self._W_future: Optional[np.ndarray] = None
        self._A: Optional[np.ndarray] = None  # canonical directions (past space)
        self._singular_values: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "CVAStateSpaceModel":
        """
        Fit CVA model on a multivariate time series.

        Parameters
        ----------
        X : np.ndarray, shape (T, n_features)
        """
        T = X.shape[0]
        total_window = self.past_window + self.future_window
        if T <= total_window:
            raise ValueError(
                f"Time series length {T} must exceed "
                f"past_window + future_window = {total_window}."
            )

        # Build Hankel matrices
        H_past = _hankel_matrix(X, self.past_window)
        H_future = _hankel_matrix(X, self.future_window)

        # Align: use overlapping rows where both windows are valid
        n_common = min(H_past.shape[0], H_future.shape[0])
        offset = self.past_window  # future starts 'past_window' steps after past
        if offset >= H_future.shape[0]:
            offset = 0
        n_use = min(H_past.shape[0] - offset, H_future.shape[0] - offset)
        if n_use <= 0:
            n_use = min(H_past.shape[0], H_future.shape[0])
            offset = 0

        Yp = H_past[offset: offset + n_use]
        Yf = H_future[: n_use]

        # Whiten
        Yp_w, self._W_past = _whiten(Yp)
        Yf_w, self._W_future = _whiten(Yf)

        # Cross-covariance SVD
        C = Yp_w.T @ Yf_w / n_use
        U, s, Vt = np.linalg.svd(C, full_matrices=False)

        k = min(self.n_components, len(s))
        self._A = self._W_past @ U[:, :k]          # (n_past_features, k)
        self._singular_values = s[:k]

        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project a time series into the CVA latent state space.

        Parameters
        ----------
        X : np.ndarray, shape (T, n_features)

        Returns
        -------
        Z : np.ndarray, shape (T - past_window + 1, n_components)
            Canonical variate scores (global state representation).
        """
        if self._A is None:
            raise RuntimeError(
                "CVAStateSpaceModel must be fitted before calling transform(). "
                "Call fit() first."
            )

        H = _hankel_matrix(X, self.past_window)
        return (H @ self._A).astype(np.float32)

    # ------------------------------------------------------------------
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    @property
    def singular_values(self) -> np.ndarray:
        if self._singular_values is None:
            raise RuntimeError(
                "CVAStateSpaceModel must be fitted before accessing singular_values. "
                "Call fit() first."
            )
        return self._singular_values
