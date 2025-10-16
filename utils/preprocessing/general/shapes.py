from __future__ import annotations
import numpy as np
from typing import Tuple

def ensure_xy_aligned(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strict alignment check: X must be 2D (n_samples, n_features),
    y must be 1D (n_samples,), and n_samples must match.
    No transposition, no truncation.
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
