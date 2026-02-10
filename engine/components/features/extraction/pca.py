from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA


def _coerce_random_state(random_state: Optional[int]) -> Optional[int]:
    """
    Ensure PCA receives only an int or None (no Generators, no hidden draws).
    """
    if random_state is None:
        return None
    if isinstance(random_state, (int, np.integer)):
        return int(random_state)
    raise TypeError(
        "random_state must be int or None. "
        "If you have a numpy Generator, derive a child *int* seed in the caller."
    )


def choose_n_components_by_variance(
    explained_variance_ratio: np.ndarray, threshold: float = 0.95
) -> int:
    """
    Choose the smallest number of components whose cumulative explained variance
    meets/exceeds `threshold` (default: 95%).
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")
    csum = np.cumsum(np.asarray(explained_variance_ratio))
    k = int(np.searchsorted(csum, threshold) + 1)
    return max(1, k)


def perform_pca(
    X: np.ndarray,
    n_components: Optional[int] = None,
    *,
    variance_threshold: float = 0.95,
    whiten: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[PCA, np.ndarray]:
    """
    Fit PCA on `X` and transform it. If `n_components` is None, determine it
    automatically via `variance_threshold` (retain that fraction of variance).

    Note: We force `svd_solver='full'` which is deterministic; `random_state`
    is accepted for API symmetry but not actually used by sklearn in this mode.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features). Got {X.shape}.")

    rs = _coerce_random_state(random_state)

    # Auto-select components by variance if not provided
    if n_components is None:
        pca_full = PCA(svd_solver="full", random_state=rs)
        pca_full.fit(X)
        n_components = choose_n_components_by_variance(
            pca_full.explained_variance_ratio_, variance_threshold
        )

    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver="full",
        random_state=rs,
    )
    X_pca = pca.fit_transform(X)
    return pca, X_pca


def pca_fit_transform_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: Optional[int] = None,
    *,
    variance_threshold: float = 0.95,
    whiten: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[PCA, np.ndarray, np.ndarray]:
    """
    Fit PCA *only on the training set* (with optional auto component selection),
    then transform both train and test.

    Note: We force `svd_solver='full'` which is deterministic; `random_state`
    is accepted for API symmetry but not actually used by sklearn in this mode.
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be 2D arrays.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test must have the same number of features.")

    rs = _coerce_random_state(random_state)

    # Auto-select n_components based on TRAIN variance if not provided
    if n_components is None:
        pca_full = PCA(svd_solver="full", random_state=rs)
        pca_full.fit(X_train)
        n_components = choose_n_components_by_variance(
            pca_full.explained_variance_ratio_, variance_threshold
        )

    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver="full",
        random_state=rs,
    )
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return pca, X_train_pca, X_test_pca
