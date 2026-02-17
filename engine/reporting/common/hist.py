from __future__ import annotations

"""Histogram + quantile primitives for reporting.

The reporting layer produces JSON-friendly payloads that are consumed by
downstream UI/API layers. Histograms show up in multiple places (decoder,
diagnostics, ensembles). This module centralizes histogram/quantile behavior
to avoid drift and duplicated edge-case handling.

Design goals
------------
* Best-effort: never raise; return None/empty payload on failure.
* JSON-friendly: lists of Python floats/ints only.
* Deterministic for the same inputs.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import math

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore

from .json_safety import safe_float_optional, safe_float_scalar


def safe_int_scalar(x: Any) -> int:
    """Make a single integer JSON-safe."""
    try:
        f = float(x)
    except Exception:
        f = float("nan")
    if not math.isfinite(f):
        return 0
    try:
        return int(f)
    except Exception:
        return 0


def safe_int_optional(x: Any) -> Optional[int]:
    """Return int(x) if it is finite and coercible, otherwise None."""
    f = safe_float_optional(x)
    if f is None:
        return None
    try:
        return int(f)
    except Exception:
        return None


def _np_mod(np_mod: Any = None):
    return np_mod if np_mod is not None else _np


def _as_finite_1d(values: Any, *, np_mod: Any = None) -> Optional[Any]:
    np = _np_mod(np_mod)
    if np is None:
        return None
    try:
        v = np.asarray(values, dtype=float).reshape(-1)
        v = v[np.isfinite(v)]
        return v
    except Exception:
        return None


def hist_init(edges: Any, *, np_mod: Any = None) -> Any:
    """Initialize histogram counts array for given edges."""
    np = _np_mod(np_mod)
    if np is None:
        return None
    e = np.asarray(edges, dtype=float)
    return np.zeros(max(0, int(e.size) - 1), dtype=float)


def hist_add_inplace(*, counts: Any, edges: Any, values: Any, np_mod: Any = None) -> None:
    """Update histogram counts in-place using fixed edges."""
    np = _np_mod(np_mod)
    if np is None or counts is None or edges is None:
        return
    try:
        vals = np.asarray(values, dtype=float).reshape(-1)
        if vals.size == 0:
            return
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return
        h, _ = np.histogram(vals, bins=np.asarray(edges, dtype=float))
        counts += h
    except Exception:
        return


def histogram_edges_counts_payload(
    values: Any,
    *,
    n_bins: int = 30,
    value_range: Optional[Tuple[float, float]] = None,
    np_mod: Any = None,
) -> Optional[Mapping[str, List[float]]]:
    """Compute a compact histogram payload {edges, counts}.

    This matches the historical behavior used by regression diagnostics.
    """

    np = _np_mod(np_mod)
    if np is None:
        return None

    v = _as_finite_1d(values, np_mod=np)
    if v is None or int(v.size) == 0:
        return None
    try:
        n_bins_i = int(n_bins) if int(n_bins) > 0 else 30

        if value_range is None:
            lo = float(np.min(v))
            hi = float(np.max(v))
        else:
            lo = float(value_range[0])
            hi = float(value_range[1])

        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        if hi <= lo:
            # degenerate: widen slightly
            eps = 1e-9 if lo == 0 else abs(lo) * 1e-9
            lo -= eps
            hi += eps

        counts, edges = np.histogram(v, bins=n_bins_i, range=(lo, hi))
        return {
            "edges": [float(x) for x in edges.tolist()],
            "counts": [float(x) for x in counts.tolist()],
        }
    except Exception:
        return None


def histogram_centers_payload(
    values: Any,
    *,
    bins: int = 30,
    np_mod: Any = None,
) -> Optional[Mapping[str, Any]]:
    """Return a compact histogram payload {x, y} using bin centers.

    This matches the historical behavior used in clustering plot payloads.
    """

    np = _np_mod(np_mod)
    if np is None:
        return None

    try:
        v = _as_finite_1d(values, np_mod=np)
        if v is None or v.size < 2:
            return None
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return None

        # adaptive binning like the original implementation
        n_bins = int(max(5, min(int(bins), int(np.floor(np.sqrt(v.size) * 2)))))
        hist, edges = np.histogram(v, bins=n_bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return {
            "x": [safe_float_scalar(x) for x in centers.tolist()],
            "y": [int(c) for c in hist.tolist()],
        }
    except Exception:
        return None


def histogram_minmax_payload(
    values: Any,
    *,
    bins: int = 20,
    np_mod: Any = None,
) -> Mapping[str, Any]:
    """Histogram payload over a data-driven [min, max] range.

    Used for pooled score histograms in ensemble reports.
    Always returns a dict with 'edges' and 'counts' keys.
    """
    np = _np_mod(np_mod)
    if np is None:
        return {"edges": [], "counts": []}

    v = _as_finite_1d(values, np_mod=np)
    if v is None or v.size == 0:
        return {"edges": [], "counts": []}

    try:
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        if np.isclose(vmin, vmax):
            eps = 1e-6 if vmax == 0.0 else abs(vmax) * 1e-6
            vmin -= eps
            vmax += eps

        edges = np.linspace(vmin, vmax, num=int(bins) + 1, dtype=float)
        counts, edges = np.histogram(v, bins=edges)
        return {
            "edges": [safe_float_scalar(x) for x in edges.tolist()],
            "counts": [safe_float_scalar(x) for x in counts.tolist()],
        }
    except Exception:
        return {"edges": [], "counts": []}


def quantiles(
    values: Any,
    *,
    qs: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    np_mod: Any = None,
) -> Optional[Mapping[str, float]]:
    """Compute quantiles for a value sequence.

    Returns a mapping like {'q05': 0.1, 'q50': 0.5, ...} or None.
    """
    np = _np_mod(np_mod)
    if np is None:
        return None
    v = _as_finite_1d(values, np_mod=np)
    if v is None or v.size == 0:
        return None
    try:
        qv = np.quantile(v, np.asarray(list(qs), dtype=float))
        out: Dict[str, float] = {}
        for q, val in zip(qs, qv.tolist()):
            key = f"q{int(round(float(q) * 100)):02d}"
            out[key] = safe_float_scalar(val)
        return out
    except Exception:
        return None


__all__ = [
    "safe_int_scalar",
    "safe_int_optional",
    "hist_init",
    "hist_add_inplace",
    "histogram_edges_counts_payload",
    "histogram_centers_payload",
    "histogram_minmax_payload",
    "quantiles",
]
