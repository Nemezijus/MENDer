from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _to_float_list(xs: Any) -> Optional[List[float]]:
    try:
        a = np.asarray(xs, dtype=float).ravel()
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
        return [float(x) for x in a.tolist()]
    except Exception:
        return None


def _merge_feature_importances(
    store: Dict[str, List[float]],
    importances: Sequence[float],
    feature_names: Optional[Sequence[str]] = None,
) -> None:
    imp = np.asarray(importances, dtype=float).ravel()
    if imp.size == 0:
        return

    if feature_names is None:
        names = [f"f{i}" for i in range(imp.size)]
    else:
        names = [str(n) for n in feature_names]
        if len(names) != imp.size:
            # fallback to indices if mismatch
            names = [f"f{i}" for i in range(imp.size)]

    for n, v in zip(names, imp.tolist()):
        if not np.isfinite(v):
            continue
        store.setdefault(n, []).append(float(v))


def _aggregate_curve_mean(curves: List[List[float]]) -> Optional[Dict[str, Any]]:
    """Align curves by shortest length and return mean + std arrays."""
    if not curves:
        return None
    lens = [len(c) for c in curves if c]
    if not lens:
        return None
    L = min(lens)
    if L <= 0:
        return None

    M = np.asarray([c[:L] for c in curves], dtype=float)
    mean = np.mean(M, axis=0)
    std = np.std(M, axis=0)
    return {
        "mean": [float(x) for x in mean.tolist()],
        "std": [float(x) for x in std.tolist()],
        "n_rounds": int(L),
    }

def _finite_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        fx = float(x)
        if not np.isfinite(fx):
            return None
        return fx
    except Exception:
        return None

__all__ = [
    "_to_float_list",
    "_merge_feature_importances",
    "_aggregate_curve_mean",
    "_finite_float_or_none",
]
