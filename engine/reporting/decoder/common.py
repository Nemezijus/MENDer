from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def as_1d(x: Any) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 0:
        return a.reshape(1)
    return a.reshape(-1)


def as_2d(x: Any) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array; got shape {a.shape}.")
    return a


def safe_float(x: Any) -> float:
    """Convert to a finite float (nan/inf become finite)."""
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        v = float(np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0))
    return float(v)


def quantile_dict(a: np.ndarray, qs: Sequence[float], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if a.size == 0:
        return out
    for q in qs:
        try:
            val = float(np.quantile(a, q))
            out[f"{prefix}_q{int(round(q * 100)):02d}"] = safe_float(val)
        except Exception:
            continue
    return out


def basic_stats(a: np.ndarray, prefix: str, qs: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}

    out: Dict[str, float] = {
        f"{prefix}_mean": safe_float(np.mean(a)),
        f"{prefix}_median": safe_float(np.median(a)),
        f"{prefix}_std": safe_float(np.std(a)),
        f"{prefix}_min": safe_float(np.min(a)),
        f"{prefix}_max": safe_float(np.max(a)),
    }
    out.update(quantile_dict(a, qs, prefix))
    return out


def normalize_proba(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    row_sums = p.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return p / row_sums


def label_to_index_map(classes: np.ndarray) -> Dict[Any, int]:
    mp: Dict[Any, int] = {}
    for i, c in enumerate(list(classes)):
        try:
            if isinstance(c, np.generic):
                c = c.item()
        except Exception:
            pass
        mp[c] = int(i)
    return mp
