from __future__ import annotations

"""Public shape/orientation utilities.

This module intentionally contains *public* helpers used across the Business Layer.
It replaces ad-hoc / private (underscore) shape coercion utilities scattered across
older modules.

Conventions
-----------
- X is 2D: (n_samples, n_features)
- y is 1D: (n_samples,)

Some loaders accept X provided as (n_features, n_samples) and will transpose when
needed to align with y.
"""

from typing import Optional, Tuple

import numpy as np


def coerce_X_only(X: np.ndarray) -> np.ndarray:
    """Basic coercion for X-only datasets (unsupervised/prediction).

    - Accepts 1D and reshapes to (n_samples, 1)
    - Enforces 2D and non-empty.
    """

    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]

    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got {X.shape}")

    n_rows, n_cols = X.shape
    if n_rows < 1 or n_cols < 1:
        raise ValueError(f"X must have at least 1 sample and 1 feature; got {X.shape}")

    return X


def align_X_y(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align X orientation to y.

    Preferred convention:
    - rows are samples, columns are features -> X.shape == (n_samples, n_features)

    If X is provided as (n_features, n_samples) and columns align with y, we transpose.
    """

    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if X.ndim == 1:
        X = X[:, None]

    if y.ndim != 1:
        raise ValueError(f"y must be 1D; got {y.shape}")

    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got {X.shape}")

    n_rows, n_cols = X.shape
    n_y = y.shape[0]

    if n_rows == n_y:
        pass
    elif n_cols == n_y:
        X = X.T
    else:
        raise ValueError(
            "Cannot align X with y: neither X.shape[0] nor X.shape[1] equals len(y). "
            f"Got X={X.shape}, len(y)={n_y}."
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"After orientation, X and y length mismatch: X.shape[0]={X.shape[0]} vs len(y)={y.shape[0]}"
        )

    return X, y


def ensure_xy_aligned(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Strict alignment check: no transposition, no truncation.

    - X must be 2D (n_samples, n_features)
    - y must be 1D (n_samples,)
    - n_samples must match
    """

    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features). Got {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (n_samples,). Got {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}.")

    return X, y


def maybe_transpose_for_expected_n_features(
    X: np.ndarray,
    *,
    expected_n_features: Optional[int],
) -> np.ndarray:
    """Best-effort transpose fix for X-only inputs.

    If expected_n_features is provided and X appears to be (n_features, n_samples)
    (i.e. X.shape[0] == expected_n_features), then transpose.
    """

    if expected_n_features is None:
        return X

    X = np.asarray(X)
    if X.ndim != 2:
        return X

    try:
        exp = int(expected_n_features)
    except Exception:
        return X

    if X.shape[1] != exp and X.shape[0] == exp:
        return X.T

    return X
