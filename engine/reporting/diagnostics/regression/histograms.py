from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

from .common import numpy


def histogram_1d(
    values: Any,
    *,
    n_bins: int = 30,
    value_range: Optional[Tuple[float, float]] = None,
) -> Optional[Mapping[str, List[float]]]:
    """Compute a compact histogram {edges, counts}.

    Returns None if it cannot be computed.
    """
    np = numpy()
    if np is None:
        return None
    try:
        v = np.asarray(values, dtype=float).reshape(-1)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        if n_bins <= 0:
            n_bins = 30

        if value_range is None:
            lo = float(np.min(v))
            hi = float(np.max(v))
        else:
            lo, hi = float(value_range[0]), float(value_range[1])

        if not np.isfinite(lo) or not np.isfinite(hi):
            return None

        if hi <= lo:
            # degenerate: widen slightly
            eps = 1e-9 if lo == 0 else abs(lo) * 1e-9
            lo -= eps
            hi += eps

        counts, edges = np.histogram(v, bins=int(n_bins), range=(lo, hi))
        return {
            "edges": [float(x) for x in edges.tolist()],
            "counts": [float(x) for x in counts.tolist()],
        }
    except Exception:
        return None
