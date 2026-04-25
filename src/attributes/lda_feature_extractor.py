"""
§3.3.2  LDA-based 1-D feature extraction per attribute.

For each attribute *k* the samples are split into two groups according to
their attribute encoding level (low ≤ 0.25  vs  high ≥ 0.75).  Linear
Discriminant Analysis is then applied to find the one-dimensional projection
that maximally separates the two groups.

The fitted projector for each attribute can later be used to transform new
data samples into the corresponding 1-D discriminant scores.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import List, Optional

from .attribute_definition import NUM_ATTRIBUTES, ATTRIBUTE_NAMES


class LDAAttributeExtractor:
    """
    Fit one LDA projector per attribute and transform data into 1-D
    discriminant scores.

    Parameters
    ----------
    n_attributes : int
        Number of attributes (defaults to ``NUM_ATTRIBUTES``).
    """

    def __init__(self, n_attributes: int = NUM_ATTRIBUTES) -> None:
        self.n_attributes = n_attributes
        self._lda_models: List[Optional[LinearDiscriminantAnalysis]] = [
            None
        ] * n_attributes
        self._fitted: bool = False

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        attribute_labels: np.ndarray,
    ) -> "LDAAttributeExtractor":
        """
        Fit one LDA model per attribute.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Raw sensor / time-series features.
        attribute_labels : np.ndarray, shape (n_samples, n_attributes)
            Attribute encoding values in {0, 0.5, 1} for every sample.
        """
        if X.shape[0] != attribute_labels.shape[0]:
            raise ValueError(
                "X and attribute_labels must have the same number of samples."
            )
        if attribute_labels.shape[1] != self.n_attributes:
            raise ValueError(
                f"Expected {self.n_attributes} attribute columns, "
                f"got {attribute_labels.shape[1]}."
            )

        for k in range(self.n_attributes):
            labels_k = attribute_labels[:, k]
            # Binarise: low (0) vs high (1); samples at 0.5 are excluded
            mask_low = labels_k <= 0.25
            mask_high = labels_k >= 0.75
            mask = mask_low | mask_high

            if mask.sum() < 2 or mask_low.sum() < 1 or mask_high.sum() < 1:
                # Not enough representative samples – skip
                continue

            X_sel = X[mask]
            y_sel = (labels_k[mask] >= 0.75).astype(int)

            lda = LinearDiscriminantAnalysis(n_components=1)
            lda.fit(X_sel, y_sel)
            self._lda_models[k] = lda

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X into a matrix of 1-D discriminant scores.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        scores : np.ndarray, shape (n_samples, n_attributes)
            Column *k* holds the 1-D LDA score for attribute *k*.
            Columns whose LDA model was not fitted (insufficient data)
            are filled with ``nan``.
        """
        if not self._fitted:
            raise RuntimeError(
                "LDAAttributeExtractor must be fitted before calling transform(). "
                "Call fit() first."
            )
        n_samples = X.shape[0]
        scores = np.full((n_samples, self.n_attributes), np.nan, dtype=np.float32)
        for k, lda in enumerate(self._lda_models):
            if lda is not None:
                scores[:, k] = lda.transform(X).ravel()
        return scores

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        X: np.ndarray,
        attribute_labels: np.ndarray,
    ) -> np.ndarray:
        """Fit and immediately transform."""
        return self.fit(X, attribute_labels).transform(X)

    # ------------------------------------------------------------------
    @property
    def attribute_names(self) -> List[str]:
        return ATTRIBUTE_NAMES[: self.n_attributes]
