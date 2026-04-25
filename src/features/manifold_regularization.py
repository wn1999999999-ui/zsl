"""
§3.3.3  Manifold-regularised dynamic analysis with Local Preserving Projection.

Locality Preserving Projection (LPP) finds a linear embedding that preserves
the neighbourhood structure of the data manifold by minimising a weighted
sum of squared distances between nearby samples.

The manifold-regularised feature extractor combines:
    1. LPP for *local* manifold preservation.
    2. An optional regularisation term that penalises the dynamic
       (temporal) variation of consecutive projected samples.

Algorithm (LPP)
---------------
1. Build a k-NN weight graph W on the input data.
2. Solve the generalised eigenproblem:
       X^T L X a = λ  X^T D X a
   where  L = D - W  is the graph Laplacian,
          D is the diagonal degree matrix.
3. Retain the eigenvectors with the *smallest* eigenvalues as the
   local-preserving projection directions.
"""

import numpy as np
from typing import Optional
from sklearn.neighbors import NearestNeighbors


class LocalPreservingProjection:
    """
    Locality Preserving Projection (LPP).

    Parameters
    ----------
    n_components : int
        Output dimensionality (number of projection directions).
    n_neighbors  : int
        Number of neighbours used to build the affinity graph (default 5).
    heat_kernel_t: float
        Bandwidth of the heat-kernel weight  w_ij = exp(-||xi-xj||²/t).
    dynamic_reg  : float
        Weight for the dynamic (temporal-smoothness) regularisation term.
        Set to 0 to disable.
    """

    def __init__(
        self,
        n_components: int = 8,
        n_neighbors: int = 5,
        heat_kernel_t: float = 1.0,
        dynamic_reg: float = 0.01,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.heat_kernel_t = heat_kernel_t
        self.dynamic_reg = dynamic_reg

        self._W_proj: Optional[np.ndarray] = None  # (n_features, n_components)

    # ------------------------------------------------------------------
    def _build_weight_matrix(self, X: np.ndarray) -> np.ndarray:
        """Build the k-NN heat-kernel weight matrix."""
        n = X.shape[0]
        k = min(self.n_neighbors, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(X)
        distances, indices = nbrs.kneighbors(X)

        W = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for idx, dist in zip(indices[i, 1:], distances[i, 1:]):
                w = np.exp(-dist ** 2 / (self.heat_kernel_t + 1e-12))
                W[i, idx] = w
                W[idx, i] = w  # symmetrise
        return W

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "LocalPreservingProjection":
        """
        Fit LPP on data matrix X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        """
        n, d = X.shape
        W = self._build_weight_matrix(X)
        D = np.diag(W.sum(axis=1))
        L = D - W  # graph Laplacian

        # Optional dynamic regularisation: penalise ||x_{t+1}p - x_t p||^2
        if self.dynamic_reg > 0.0:
            # Build a temporal-difference Laplacian
            diff_mat = np.zeros((n, n), dtype=np.float64)
            for i in range(n - 1):
                diff_mat[i, i] += 1.0
                diff_mat[i, i + 1] -= 1.0
                diff_mat[i + 1, i] -= 1.0
                diff_mat[i + 1, i + 1] += 1.0
            L = L + self.dynamic_reg * diff_mat

        # Generalised eigenproblem:  X^T L X a = λ X^T D X a
        XtLX = X.T @ L @ X
        XtDX = X.T @ D @ X

        # Regularise XtDX for numerical stability
        XtDX += np.eye(d) * 1e-9

        try:
            eigvals, eigvecs = np.linalg.eig(np.linalg.solve(XtDX, XtLX))
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(XtLX)

        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        # Select eigenvectors with smallest eigenvalues (most locality-preserving)
        order = np.argsort(eigvals)
        k = min(self.n_components, d)
        self._W_proj = eigvecs[:, order[:k]].astype(np.float32)

        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X into the local-preserving subspace.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_components)
        """
        if self._W_proj is None:
            raise RuntimeError(
                "LocalPreservingProjection must be fitted before calling transform(). "
                "Call fit() first."
            )
        return (X @ self._W_proj).astype(np.float32)

    # ------------------------------------------------------------------
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
