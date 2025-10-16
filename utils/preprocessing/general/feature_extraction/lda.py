from __future__ import annotations

from typing import Optional, Tuple, Sequence, Union, Literal
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


SolverName = Literal["svd", "lsqr", "eigen"]


def _coerce_xy(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features). Got {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (n_samples,). Got {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}.")
    return X, y


def _cap_n_components(
    n_components: Optional[int],
    *,
    n_features: int,
    n_classes: int,
) -> int:
    """
    LDA subspace has dimension at most min(n_features, n_classes - 1).
    If n_components is None, use that maximum; otherwise clamp into [1, max_dim].
    """
    if n_classes < 2:
        raise ValueError("LDA requires at least two classes.")

    max_dim = min(n_features, n_classes - 1)
    if max_dim < 1:
        # This can only happen when n_classes == 1, which we already guard against.
        raise ValueError("Cannot form an LDA subspace with <2 classes.")

    if n_components is None:
        return max_dim

    if not isinstance(n_components, (int, np.integer)):
        raise TypeError("n_components must be an int or None.")
    k = int(n_components)
    if k < 1:
        raise ValueError("n_components must be >= 1.")
    return min(k, max_dim)


def perform_lda(
    X: np.ndarray,
    y: np.ndarray,
    n_components: Optional[int] = None,
    *,
    solver: SolverName = "svd",
    shrinkage: Optional[Union[float, Literal["auto"]]] = None,
    priors: Optional[Sequence[float]] = None,
    tol: float = 1e-4,
) -> Tuple[LinearDiscriminantAnalysis, np.ndarray]:
    """
    Supervised projection via Linear Discriminant Analysis (LDA).
    Fits on (X, y) and returns (fitted LDA, X_lda).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    n_components : desired subspace dimension; if None, uses min(n_features, n_classes-1)
    solver : {'svd','lsqr','eigen'}
        - 'svd': default, robust; ignores 'shrinkage'
        - 'lsqr'/'eigen': support shrinkage
    shrinkage : {None, 'auto', float}
        Only used with 'lsqr' or 'eigen'. See sklearn docs.
    priors : class prior probabilities; shape (n_classes,), optional
    tol : convergence tolerance for 'svd' solver

    Returns
    -------
    lda : fitted LinearDiscriminantAnalysis instance
    X_lda : transformed data, shape (n_samples, n_components_eff)
    """
    X, y = _coerce_xy(X, y)
    classes = np.unique(y)
    k = _cap_n_components(n_components, n_features=X.shape[1], n_classes=classes.size)

    lda = LinearDiscriminantAnalysis(
        n_components=k,
        solver=solver,
        shrinkage=shrinkage,
        priors=None if priors is None else np.asarray(priors, dtype=float),
        tol=tol,
        store_covariance=False,  # can be toggled later if needed
    )
    X_lda = lda.fit_transform(X, y)
    return lda, X_lda


def lda_fit_transform_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    n_components: Optional[int] = None,
    *,
    solver: SolverName = "svd",
    shrinkage: Optional[Union[float, Literal["auto"]]] = None,
    priors: Optional[Sequence[float]] = None,
    tol: float = 1e-4,
) -> Tuple[LinearDiscriminantAnalysis, np.ndarray, np.ndarray]:
    """
    Fit LDA on the TRAIN set only, then transform both train and test.

    Parameters
    ----------
    X_train, X_test : arrays with the same number of features (2D)
    y_train : labels for training samples (1D)
    n_components, solver, shrinkage, priors, tol : see perform_lda

    Returns
    -------
    lda : fitted LinearDiscriminantAnalysis instance (trained on X_train, y_train)
    X_train_lda : transformed train data, shape (n_train, n_components_eff)
    X_test_lda  : transformed test data,  shape (n_test,  n_components_eff)
    """
    X_train, y_train = _coerce_xy(X_train, y_train)
    X_test = np.asarray(X_test)
    if X_test.ndim != 2:
        raise ValueError(f"X_test must be 2D. Got {X_test.shape}.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test must have the same number of features.")

    classes = np.unique(y_train)
    k = _cap_n_components(n_components, n_features=X_train.shape[1], n_classes=classes.size)

    lda = LinearDiscriminantAnalysis(
        n_components=k,
        solver=solver,
        shrinkage=shrinkage,
        priors=None if priors is None else np.asarray(priors, dtype=float),
        tol=tol,
        store_covariance=False,
    )
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    return lda, X_train_lda, X_test_lda
