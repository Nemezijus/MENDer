from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

from engine.reporting.common.json_safety import ReportError, add_report_error

from .common import numpy


def downsample_xy(
    x: Any,
    y: Any,
    *,
    max_points: int = 5000,
    seed: int = 0,
) -> Optional[Mapping[str, List[float]]]:
    """Return a downsampled {x, y} payload for scatter-like plots."""
    np = numpy()
    if np is None:
        return None
    try:
        xa = np.asarray(x, dtype=float).reshape(-1)
        ya = np.asarray(y, dtype=float).reshape(-1)
        if xa.size == 0 or ya.size == 0 or xa.size != ya.size:
            return None
        m = np.isfinite(xa) & np.isfinite(ya)
        xa = xa[m]
        ya = ya[m]
        n = int(xa.size)
        if n == 0:
            return None
        if max_points is None or max_points <= 0 or n <= max_points:
            return {"x": [float(v) for v in xa.tolist()], "y": [float(v) for v in ya.tolist()]}

        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n, size=int(max_points), replace=False)
        xs = xa[idx]
        ys = ya[idx]
        return {"x": [float(v) for v in xs.tolist()], "y": [float(v) for v in ys.tolist()]}
    except Exception:
        return None


def binned_error_by_true(
    y_true: Any,
    y_pred: Any,
    *,
    n_bins: int = 10,
) -> Optional[Mapping[str, List[float]]]:
    """Bin by y_true quantiles and compute MAE/RMSE per bin."""
    np = numpy()
    if np is None:
        return None
    try:
        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_pred, dtype=float).reshape(-1)
        if yt.size == 0 or yp.size == 0 or yt.size != yp.size:
            return None
        m = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[m]
        yp = yp[m]
        n = int(yt.size)
        if n == 0:
            return None
        if n_bins <= 0:
            n_bins = 10

        qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
        edges = np.quantile(yt, qs)

        mae: List[float] = []
        rmse: List[float] = []
        counts: List[float] = []

        for b in range(int(n_bins)):
            lo = float(edges[b])
            hi = float(edges[b + 1])
            if b == int(n_bins) - 1:
                mb = (yt >= lo) & (yt <= hi)
            else:
                mb = (yt >= lo) & (yt < hi)
            nb = int(np.sum(mb))
            counts.append(float(nb))
            if nb == 0:
                mae.append(0.0)
                rmse.append(0.0)
                continue
            e = yp[mb] - yt[mb]
            mae.append(float(np.mean(np.abs(e))))
            rmse.append(float(np.sqrt(np.mean(e * e))))

        return {
            "edges": [float(x) for x in edges.tolist()],
            "mae": mae,
            "rmse": rmse,
            "n": counts,
        }
    except Exception:
        return None