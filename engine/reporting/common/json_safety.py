from __future__ import annotations

"""JSON-safety helpers.

These helpers are used by reporting/payload builders to ensure response
objects contain only finite JSON-serializable numbers.

Policy
------
* NaN      -> 0.0
* +inf     -> 1.0
* -inf     -> 0.0
"""

from typing import Any, Iterable, List, Optional
import math

import numpy as np


def safe_float_list(arr: Any) -> List[float]:
    """Convert array-like to a JSON-safe list of finite floats."""

    a = np.asarray(arr, dtype=float)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    return a.tolist()


def safe_float_scalar(x: Any) -> float:
    """Make a single float JSON-safe (no NaN/inf)."""

    try:
        f = float(x)
    except Exception:
        f = float("nan")
    return float(np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=0.0))


def safe_float_optional(x: Any) -> Optional[float]:
    """Return float(x) if it's finite, otherwise None."""

    try:
        f = float(x)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def dedupe_preserve_order(seq: Iterable[Any]) -> List[Any]:
    """Return list with duplicates removed while preserving order."""

    out: List[Any] = []
    seen = set()
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out
