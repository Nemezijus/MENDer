from __future__ import annotations

"""Lightweight helpers for best-effort error/warning recording in reporting payloads.

Reporting should never raise, but silent failures make regressions hard to debug.
These helpers standardize how we attach structured errors (and optional warnings)
into JSON-friendly payload dicts.

The UI does not need to surface these yet; they exist primarily for debugging and
future observability.
"""

from typing import Any, Dict, List, Optional

from .json_safety import ReportError, add_report_error


def ensure_errors(payload: Dict[str, Any]) -> List[ReportError]:
    """Ensure payload has an ``errors`` list and return it."""
    try:
        v = payload.get("errors")
        if isinstance(v, list):
            return v  # type: ignore[return-value]
    except Exception:
        pass

    errs: List[ReportError] = []
    try:
        payload["errors"] = errs
    except Exception:
        # If payload is not mutable, just return a private list.
        return errs
    return errs


def record_error(
    payload: Dict[str, Any],
    *,
    where: str,
    exc: BaseException,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a structured error marker into ``payload['errors']`` (never raises)."""
    try:
        add_report_error(ensure_errors(payload), where=where, exc=exc, context=context)
    except Exception:
        pass


def ensure_warnings(payload: Dict[str, Any]) -> List[str]:
    """Ensure payload has a ``warnings`` list and return it."""
    try:
        v = payload.get("warnings")
        if isinstance(v, list):
            return v  # type: ignore[return-value]
    except Exception:
        pass

    ws: List[str] = []
    try:
        payload["warnings"] = ws
    except Exception:
        return ws
    return ws


def record_warning(payload: Dict[str, Any], msg: str) -> None:
    """Append a warning message into ``payload['warnings']`` (never raises)."""
    try:
        ensure_warnings(payload).append(str(msg))
    except Exception:
        pass
