# utils/preprocessing/feature_scaling.py
from __future__ import annotations

from typing import Literal, Optional, Tuple
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    QuantileTransformer,
)

ScalerName = Literal["standard", "minmax", "robust", "maxabs", "quantile", "none"]


def _make_scaler(
    method: ScalerName,
    *,
    quantile_output_distribution: Literal["uniform", "normal"] = "normal",
) -> Optional[object]:
    """
    Factory producing a scikit-learn scaler instance (or None if 'none').
    """
    method = method.lower()
    if method == "standard":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    if method == "robust":
        return RobustScaler()  # robust to outliers (uses IQR)
    if method == "maxabs":
        return MaxAbsScaler()  # good for sparse, preserves sparsity
    if method == "quantile":
        # Nonlinear; maps marginals to chosen distribution (can squash outliers)
        return QuantileTransformer(
            output_distribution=quantile_output_distribution,
            copy=True,
            subsample=int(1e5),
            random_state=0,
        )
    if method == "none":
        return None
    raise ValueError(f"Unknown scaling method '{method}'.")


def scale_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    method: ScalerName = "standard",
    quantile_output_distribution: Literal["uniform", "normal"] = "normal",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a scaler on train, apply to both train and test. Prevents test leakage.

    Parameters
    ----------
    X_train : array-like of shape (n_samples_train, n_features)
    X_test : array-like of shape (n_samples_test, n_features)
    method : {'standard','minmax','robust','maxabs','quantile','none'}, default='standard'
        Which scaling strategy to use.
    quantile_output_distribution : {'uniform','normal'}, default='normal'
        Only used when method='quantile'.

    Returns
    -------
    X_train_scaled, X_test_scaled : ndarray
        Scaled copies of the inputs (or originals if method='none').

    Notes
    -----
    - Always fits the scaler **only on the training data**, then transforms
      both train and test.
    - For 'quantile', this is a nonlinear transform that can affect model
      interpretability; use intentionally.
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be 2D.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test must have the same number of features.")

    scaler = _make_scaler(method, quantile_output_distribution=quantile_output_distribution)
    if scaler is None:
        # No scaling requested
        return X_train.copy(), X_test.copy()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
