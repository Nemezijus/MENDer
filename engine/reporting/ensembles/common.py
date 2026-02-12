from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np

from engine.reporting.common.json_safety import ReportError, add_report_error

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

def attach_report_error(target: Any, *, where: str, exc: BaseException, context: dict[str, Any] | None = None) -> None:
    """Attach a structured error marker to an accumulator/payload object."""
    try:
        errors = getattr(target, "_errors", None)
        if errors is None:
            errors = []
            setattr(target, "_errors", errors)
        add_report_error(errors, where=where, exc=exc, context=context)
    except Exception:
        pass
