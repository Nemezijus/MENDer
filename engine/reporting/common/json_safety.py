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

from typing import Any, Dict, Iterable, List, Optional, TypedDict
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


class ReportError(TypedDict, total=False):
    """Lightweight, JSON-friendly error marker for reporting layers."""

    where: str
    error: str
    error_type: str
    context: Dict[str, Any]


def add_report_error(
    errors: List[ReportError],
    *,
    where: str,
    exc: BaseException,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a structured error marker (best-effort, never raises)."""
    try:
        errors.append(
            {
                "where": str(where),
                "error": str(exc),
                "error_type": type(exc).__name__,
                "context": dict(context) if context else {},
            }
        )
    except Exception:
        # Reporting must not fail due to error formatting.
        pass


def error_row(*, where: str, exc: BaseException, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a single-row table-friendly error marker."""
    err: List[ReportError] = []
    add_report_error(err, where=where, exc=exc, context=context)
    return {"__error__": True, "errors": err}


def error_row_from_errors(errors: List[ReportError]) -> Dict[str, Any]:
    """Create a table-friendly error marker row from an existing error list."""
    try:
        return {"__error__": True, "errors": list(errors)}
    except Exception:
        return {"__error__": True, "errors": []}
