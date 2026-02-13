from __future__ import annotations

"""Adapters for converting internal SearchCV results to JSON-ready values."""

import math
from typing import Any, Dict, List, Sequence

import numpy as np


def to_py(v: Any) -> Any:
    """Convert numpy scalars/arrays into pure python types for JSON."""

    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def sanitize_float_list(values: Sequence[float]) -> List[float | None]:
    out: List[float | None] = []
    for v in values:
        try:
            fv = float(v)
            out.append(fv if math.isfinite(fv) else None)
        except Exception:
            out.append(None)
    return out


def build_cv_results(
    *,
    param_keys: Sequence[str],
    params_list: Sequence[Dict[str, Any]],
    mean_train: Sequence[float],
    std_train: Sequence[float],
    mean_val: Sequence[float],
    std_val: Sequence[float],
) -> Dict[str, Any]:
    """Build the cv_results_ dict expected by the frontend."""

    cv_results: Dict[str, List[Any]] = {}
    for k in param_keys:
        cv_results[f"param_{k}"] = [to_py(p.get(k)) for p in params_list]

    cv_results["mean_score"] = sanitize_float_list(mean_train)
    cv_results["std_score"] = sanitize_float_list(std_train)
    cv_results["mean_test_score"] = sanitize_float_list(mean_val)
    cv_results["std_test_score"] = sanitize_float_list(std_val)
    return cv_results
