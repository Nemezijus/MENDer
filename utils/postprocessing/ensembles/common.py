from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np

def _safe_float(x: Any) -> float:
    try:
        return float(np.nan_to_num(float(x), nan=0.0, posinf=1.0, neginf=0.0))
    except Exception:
        return 0.0


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    a = np.asarray(xs, dtype=float)
    return float(np.mean(a)), float(np.std(a))