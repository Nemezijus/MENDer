from __future__ import annotations

from typing import Any, Optional
import numpy as np


def fit_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
) -> Any:
    """
    Fit a scikit-learn–style estimator on training data.

    Parameters
    ----------
    model : Any
        Estimator exposing `fit(X, y, **kwargs)`.
    X_train : array-like of shape (n_samples, n_features)
    y_train : array-like of shape (n_samples,)
    sample_weight : array-like of shape (n_samples,), optional
        Per-sample weights; only passed if not None.

    Returns
    -------
    model : Any
        The same estimator, after fitting (standard sklearn behavior).

    Raises
    ------
    AttributeError
        If `model` does not have a `fit` method.
    ValueError
        If input shapes are inconsistent.
    """
    if not hasattr(model, "fit"):
        raise AttributeError("`model` has no `.fit(...)` method.")

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()

    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D; got {X_train.shape}.")
    if y_train.ndim != 1:
        raise ValueError(f"y_train must be 1D; got {y_train.shape}.")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train and y_train length mismatch: {X_train.shape[0]} vs {y_train.shape[0]}."
        )

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight).ravel()
        if sample_weight.shape[0] != y_train.shape[0]:
            raise ValueError(
                "sample_weight must match the number of training samples."
            )
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    return model
