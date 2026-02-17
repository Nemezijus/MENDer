from __future__ import annotations

from typing import Any, Mapping, Optional

from .deps import np


def histogram_payload(values: Any, *, bins: int = 30) -> Optional[Mapping[str, Any]]:
    """Return a small histogram payload {x, y} for JSON transport."""
    if np is None:
        return None
    try:
        v = np.asarray(values, dtype=float).reshape(-1)
        v = v[np.isfinite(v)]
        if v.size < 2:
            return None
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return None

        n_bins = int(max(5, min(int(bins), int(np.floor(np.sqrt(v.size) * 2)))))
        hist, edges = np.histogram(v, bins=n_bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return {
            "x": [float(x) for x in centers.tolist()],
            "y": [int(c) for c in hist.tolist()],
        }
    except Exception:
        return None
