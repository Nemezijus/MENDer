from __future__ import annotations

from typing import Any, Optional, Tuple

import math

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


def numpy() -> Any:
    return np


def as_1d(x: Any) -> Any:
    if np is None:
        return x
    a = np.asarray(x)
    if a.ndim == 0:
        return a.reshape(1)
    return a.reshape(-1)


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return float(v)


def maybe_corr(x: Any, y: Any) -> Tuple[Optional[float], Optional[float]]:
    """Return (pearson_r, spearman_r) best-effort."""
    if np is None:
        return None, None

    try:
        xa = np.asarray(x, dtype=float).reshape(-1)
        ya = np.asarray(y, dtype=float).reshape(-1)
    except Exception:
        return None, None

    if xa.size == 0 or ya.size == 0 or xa.size != ya.size:
        return None, None

    m = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[m]
    ya = ya[m]
    if xa.size < 2:
        return None, None

    # Pearson
    try:
        pearson = float(np.corrcoef(xa, ya)[0, 1])
        if not np.isfinite(pearson):
            pearson = None
    except Exception:
        pearson = None

    # Spearman: rank then Pearson
    try:
        rx = xa.argsort().argsort().astype(float)
        ry = ya.argsort().argsort().astype(float)
        spearman = float(np.corrcoef(rx, ry)[0, 1])
        if not np.isfinite(spearman):
            spearman = None
    except Exception:
        spearman = None

    return pearson, spearman
