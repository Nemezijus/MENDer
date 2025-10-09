# utils/preprocessing/general/pca_utils.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def choose_n_components_by_variance(
    explained_variance_ratio: np.ndarray, threshold: float = 0.95
) -> int:
    """
    Choose the smallest number of components whose cumulative explained variance
    meets/exceeds `threshold` (default: 95%).

    Parameters
    ----------
    explained_variance_ratio : (n_features,) array
        PCA.explained_variance_ratio_ from a preliminary PCA (all components).
    threshold : float, default=0.95
        Target cumulative variance to retain (0 < threshold <= 1).

    Returns
    -------
    k : int
        Number of components to keep.
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
    standardize: bool = True,
) -> Tuple[PCA, np.ndarray, Optional[StandardScaler]]:
    """
    Fit PCA on `X` and transform it. If `n_components` is None, determine it
    automatically via `variance_threshold` (retain that fraction of variance).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input matrix (any features).
    n_components : int or None, default=None
        If None, auto-pick components by cumulative variance threshold.
        If provided, use exactly this many components.
    variance_threshold : float, default=0.95
        Target cumulative variance to retain when auto-selecting components.
    whiten : bool, default=False
        If True, whiten principal components (scales PCs to unit variance).
    random_state : int or None, default=None
        Random state forwarded to PCA (for deterministic SVD ordering when tied).
    standardize : bool, default=True
        If True, z-score features before PCA (fit on full X in this function).

    Returns
    -------
    pca : sklearn.decomposition.PCA
        Fitted PCA object.
    X_pca : ndarray of shape (n_samples, n_components)
        Transformed data in PC space.
    scaler : StandardScaler or None
        Fitted scaler if `standardize=True`, else None.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features). Got {X.shape}.")

    scaler = None
    X_proc = X
    if standardize:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X)

    # Auto-select components by variance if not provided
    if n_components is None:
        # Preliminary PCA (full rank or min(n_samples, n_features))
        pca_full = PCA(svd_solver="full", random_state=random_state)
        pca_full.fit(X_proc)
        k = choose_n_components_by_variance(
            pca_full.explained_variance_ratio_, variance_threshold
        )
        n_components = k

    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver="full",
        random_state=random_state,
    )
    X_pca = pca.fit_transform(X_proc)
    return pca, X_pca, scaler


def pca_fit_transform_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: Optional[int] = None,
    *,
    variance_threshold: float = 0.95,
    whiten: bool = False,
    random_state: Optional[int] = None,
    standardize: bool = False,
) -> Tuple[PCA, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    Fit PCA *only on the training set* (with optional auto component selection),
    then transform both train and test.

    This mirrors the usage pattern in your book example.

    Parameters
    ----------
    X_train, X_test : arrays
        Train and test matrices with same number of features.
    n_components : int or None
        If None, auto-pick by `variance_threshold`. Else, fixed.
    variance_threshold : float, default=0.95
        Cumulative variance target for auto component selection.
    whiten : bool, default=False
        Whiten PCs.
    random_state : int or None
        Random state forwarded to PCA.
    standardize : bool, default=True
        Z-score features using *train* statistics, apply to both sets.

    Returns
    -------
    pca : PCA
        Fitted PCA object (trained on X_train).
    X_train_pca : ndarray
        Transformed train set.
    X_test_pca : ndarray
        Transformed test set.
    scaler : StandardScaler or None
        Fitted scaler (train-only), or None if `standardize=False`.
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be 2D arrays.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test must have the same number of features.")

    scaler = None
    Xtr = X_train
    Xte = X_test
    if standardize:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)

    # Auto-select n_components based on TRAIN variance if not provided
    if n_components is None:
        pca_full = PCA(svd_solver="full", random_state=random_state)
        pca_full.fit(Xtr)
        k = choose_n_components_by_variance(
            pca_full.explained_variance_ratio_, variance_threshold
        )
        n_components = k

    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver="full",
        random_state=random_state,
    )
    X_train_pca = pca.fit_transform(Xtr)
    X_test_pca = pca.transform(Xte)
    return pca, X_train_pca, X_test_pca, scaler
