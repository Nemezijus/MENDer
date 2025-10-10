from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA


def choose_n_components_by_variance(
    explained_variance_ratio: np.ndarray, threshold: float = 0.95
) -> int:
    """
    Choose the smallest number of components whose cumulative explained variance
    meets/exceeds `threshold` (default: 95%).

    Parameters
    ----------
    explained_variance_ratio : (n_components_total,) array
        PCA.explained_variance_ratio_ from a PCA fit (typically with all comps).
    threshold : float, default=0.95
        Target cumulative variance to retain (0 < threshold <= 1).

    Returns
    -------
    k : int
        Number of components to keep (>= 1).
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")
    csum = np.cumsum(explained_variance_ratio)
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

    NOTE
    ----
    This function does NOT scale/standardize. If you want z-scoring or other
    scaling, do it upstream and pass the processed X here.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input matrix (any features).
    n_components : int or None, default=None
        If None, auto-pick components by cumulative variance threshold using
        a preliminary PCA fit on X. If provided, use exactly this many.
    variance_threshold : float, default=0.95
        Target cumulative variance to retain when auto-selecting components.
    whiten : bool, default=False
        If True, whiten principal components (unit-variance PCs).
    random_state : int or None, default=None
        Random state for deterministic SVD ordering when tied.

    Returns
    -------
    pca : sklearn.decomposition.PCA
        Fitted PCA object.
    X_pca : ndarray of shape (n_samples, n_components)
        Transformed data in PC space.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features). Got {X.shape}.")

    # Auto-select components by variance if not provided
    if n_components is None:
        pca_full = PCA(svd_solver="full", random_state=random_state)
        pca_full.fit(X)
        n_components = choose_n_components_by_variance(
            pca_full.explained_variance_ratio_, variance_threshold
        )

    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver="full",
        random_state=random_state,
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

    NOTE
    ----
    This function does NOT scale/standardize. If you want z-scoring or other
    scaling, do it upstream (fit on train, apply to test), then call this.

    Parameters
    ----------
    X_train, X_test : arrays
        Train and test matrices with the same number of features.
    n_components : int or None
        If None, auto-pick by `variance_threshold` using a preliminary PCA on
        X_train. Else, fixed number of components.
    variance_threshold : float, default=0.95
        Cumulative variance target for auto component selection (train-only).
    whiten : bool, default=False
        Whiten PCs.
    random_state : int or None
        Random state forwarded to PCA.

    Returns
    -------
    pca : PCA
        Fitted PCA object (trained on X_train).
    X_train_pca : ndarray
        Transformed train set.
    X_test_pca : ndarray
        Transformed test set.
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be 2D arrays.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test must have the same number of features.")

    # Auto-select n_components based on TRAIN variance if not provided
    if n_components is None:
        pca_full = PCA(svd_solver="full", random_state=random_state)
        pca_full.fit(X_train)
        n_components = choose_n_components_by_variance(
            pca_full.explained_variance_ratio_, variance_threshold
        )

    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver="full",
        random_state=random_state,
    )
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return pca, X_train_pca, X_test_pca
