from __future__ import annotations

"""Dataset inspection use-case.

This use-case converts already-loaded arrays into a UI-friendly inspection
payload. It intentionally avoids any file IO; boundaries (backend/scripts)
should load arrays using appropriate adapters and then call this.
"""

from typing import Any, Optional

import numpy as np

from engine.reporting.diagnostics.dataset_inspection import build_inspection_payload


def inspect_dataset(
    *,
    X: Any,
    y: Optional[Any] = None,
    treat_missing_y_as_unsupervised: bool = False,
) -> dict:
    """Inspect arrays and return a JSON-serializable payload."""

    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape={getattr(X_arr, 'shape', None)}")

    y_arr = None if y is None else np.asarray(y).ravel()

    return build_inspection_payload(
        X=X_arr,
        y=y_arr,
        treat_missing_y_as_unsupervised=bool(treat_missing_y_as_unsupervised),
    )
