"""Array loading helpers for backend boundary (delegates parsing to Engine)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from engine.api import load_from_data_model

from .data_config import build_data_config
from .errors import LoadError


def load_X_optional_y(
    npz_path: Optional[str],
    x_key: Optional[str],
    y_key: Optional[str],
    x_path: Optional[str],
    y_path: Optional[str],
    expected_n_features: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Prediction/inspect helper: X is required, y is optional.

    Notes
    -----
    - Parsing is delegated to Engine readers via the stable API
      (:func:`engine.api.load_from_data_model`).
    - If expected_n_features is provided, we apply a best-effort transpose fix for
      X-only inputs when the data appears to be (n_features, n_samples).
    """
    try:
        cfg = build_data_config(npz_path, x_key, y_key, x_path, y_path)

        X, y, _feature_names = load_from_data_model(cfg)

        if expected_n_features is not None:
            try:
                exp = int(expected_n_features)
                if X.ndim == 2 and X.shape[1] != exp and X.shape[0] == exp:
                    X = X.T
            except Exception:
                pass

        return X, y

    except LoadError:
        raise
    except Exception as e:
        raise LoadError(str(e))
