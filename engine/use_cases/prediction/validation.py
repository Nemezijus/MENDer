from __future__ import annotations

from typing import Optional

import numpy as np


def maybe_validate_n_features(X_arr: np.ndarray, n_features_expected: Optional[int]) -> None:
    """Validate that X has the expected number of features if known."""

    if n_features_expected is None:
        return
    exp = int(n_features_expected)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D X for prediction; got shape {X_arr.shape}.")
    if int(X_arr.shape[1]) != exp:
        raise ValueError(
            f"Feature mismatch: model expects {exp} features, but X has shape {X_arr.shape}."
        )
