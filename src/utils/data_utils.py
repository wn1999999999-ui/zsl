"""
§3.3.3  Shared data utilities.

Provides a standard 8:2 stratified train/test split and a simple synthetic
data generator for unit-testing and demonstration purposes.
"""

import numpy as np
from sklearn.model_selection import train_test_split as _sklearn_split
from typing import Tuple


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training (80 %) and test (20 %) subsets.

    Parameters
    ----------
    X            : feature matrix, shape (n_samples, n_features).
    y            : integer class labels, shape (n_samples,).
    test_size    : fraction of data reserved for testing (default 0.2).
    random_state : reproducibility seed.
    stratify     : maintain class proportions if True (default).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    strat = y if stratify else None
    return _sklearn_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )


def generate_synthetic_flow_data(
    n_samples_per_class: int = 200,
    n_features: int = 32,
    n_classes: int = 6,
    noise_std: float = 0.5,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic multivariate time-series data for *n_classes* flow
    regimes with Gaussian class-conditional distributions.

    Returns
    -------
    X : np.ndarray, shape (n_samples_per_class * n_classes, n_features)
    y : np.ndarray, shape (n_samples_per_class * n_classes,)
    """
    rng = np.random.RandomState(random_state)
    X_list, y_list = [], []
    for c in range(n_classes):
        mean = rng.randn(n_features)
        cov = np.eye(n_features) * (noise_std ** 2)
        samples = rng.multivariate_normal(mean, cov, size=n_samples_per_class)
        X_list.append(samples.astype(np.float32))
        y_list.append(np.full(n_samples_per_class, c, dtype=np.int64))
    return np.vstack(X_list), np.concatenate(y_list)
