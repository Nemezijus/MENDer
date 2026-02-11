from __future__ import annotations

"""Best-effort coercion utilities used by prediction/decoder reporting.

Reporting is allowed to be defensive. These helpers try to coerce inputs to
numpy arrays when numpy is available, and otherwise fall back gracefully.
"""

from typing import Any, Optional

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


def to_1d_array(x: Any) -> Any:
    """Best-effort coercion of input to a 1D numpy array."""
    if np is None:
        return x
    try:
        arr = np.asarray(x)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr
    except Exception:
        return x


def to_2d_array(x: Any) -> Any:
    """Best-effort coercion of input to a 2D numpy array."""
    if np is None:
        return x
    try:
        arr = np.asarray(x)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr
    except Exception:
        return x


def slice_first_n(seq: Any, n: Optional[int]) -> Any:
    if n is None:
        return seq
    if np is not None and isinstance(seq, np.ndarray):
        return seq[:n]
    try:
        return seq[:n]
    except Exception:
        return seq


def numpy() -> Any:
    """Expose numpy for callers that want to branch on availability."""
    return np
