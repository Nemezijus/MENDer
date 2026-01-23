from __future__ import annotations

"""Regression diagnostics (business-logic only).

This module turns (y_true, y_pred) into compact, JSON-friendly diagnostics
that the backend can include in Train/Ensemble responses and the frontend can
plot.

It intentionally avoids any backend/frontend imports. The outputs are plain
Python types: dict, list, float, int.

The module is defensive:
- If numpy or sklearn are missing, it degrades gracefully.
- NaNs/infs are filtered where practical.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import math

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from sklearn.metrics import (  # type: ignore
        mean_absolute_error as _sk_mae,
        mean_squared_error as _sk_mse,
        median_absolute_error as _sk_med_ae,
        r2_score as _sk_r2,
        explained_variance_score as _sk_ev,
    )
except Exception:  # pragma: no cover
    _sk_mae = _sk_mse = _sk_med_ae = _sk_r2 = _sk_ev = None


def _as_1d(x: Any) -> Any:
    if np is None:
        return x
    a = np.asarray(x)
    if a.ndim == 0:
        return a.reshape(1)
    return a.reshape(-1)


def _finite_mask(a: Any) -> Any:
    if np is None:
        return None
    a = np.asarray(a, dtype=float)
    return np.isfinite(a)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return float(v)


def _maybe_corr(x: Any, y: Any) -> Tuple[Optional[float], Optional[float]]:
    """Return (pearson_r, spearman_r) best-effort."""
    if np is None:
        return None, None

    try:
        xa = np.asarray(x, dtype=float).reshape(-1)
        ya = np.asarray(y, dtype=float).reshape(-1)
    except Exception:
        return None, None

    if xa.size == 0 or ya.size == 0 or xa.size != ya.size:
        return None, None

    m = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[m]
    ya = ya[m]
    if xa.size < 2:
        return None, None

    # Pearson
    try:
        pearson = float(np.corrcoef(xa, ya)[0, 1])
        if not np.isfinite(pearson):
            pearson = None
    except Exception:
        pearson = None

    # Spearman: rank then Pearson
    try:
        rx = xa.argsort().argsort().astype(float)
        ry = ya.argsort().argsort().astype(float)
        spearman = float(np.corrcoef(rx, ry)[0, 1])
        if not np.isfinite(spearman):
            spearman = None
    except Exception:
        spearman = None

    return pearson, spearman


def histogram_1d(
    values: Any,
    *,
    n_bins: int = 30,
    value_range: Optional[Tuple[float, float]] = None,
) -> Optional[Mapping[str, List[float]]]:
    """Compute a compact histogram {edges, counts}.

    Returns None if it cannot be computed.
    """
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


def downsample_xy(
    x: Any,
    y: Any,
    *,
    max_points: int = 5000,
    seed: int = 0,
) -> Optional[Mapping[str, List[float]]]:
    """Return a downsampled {x, y} payload for scatter-like plots."""
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
    """Bin by y_true quantiles and compute MAE/RMSE per bin.

    Returns payload with:
      - edges: bin edges (quantiles)
      - mae: per-bin MAE
      - rmse: per-bin RMSE
      - n: per-bin sample counts
    """
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
        if yt.size == 0:
            return None
        nb = int(n_bins) if int(n_bins) > 1 else 10

        qs = np.linspace(0.0, 1.0, nb + 1)
        edges = np.quantile(yt, qs)

        # ensure strictly increasing edges (fallback to min/max uniform if degenerate)
        if np.allclose(edges, edges[0]):
            lo = float(np.min(yt))
            hi = float(np.max(yt))
            if hi <= lo:
                eps = 1e-9 if lo == 0 else abs(lo) * 1e-9
                lo -= eps
                hi += eps
            edges = np.linspace(lo, hi, nb + 1)

        mae_list: List[float] = []
        rmse_list: List[float] = []
        n_list: List[float] = []

        for i in range(nb):
            lo = edges[i]
            hi = edges[i + 1]
            if i < nb - 1:
                sel = (yt >= lo) & (yt < hi)
            else:
                sel = (yt >= lo) & (yt <= hi)
            if not np.any(sel):
                mae_list.append(float("nan"))
                rmse_list.append(float("nan"))
                n_list.append(0.0)
                continue
            r = yt[sel] - yp[sel]
            mae_list.append(float(np.mean(np.abs(r))))
            rmse_list.append(float(np.sqrt(np.mean(r * r))))
            n_list.append(float(np.sum(sel)))

        return {
            "edges": [float(v) for v in edges.tolist()],
            "mae": [float(v) if np.isfinite(v) else float("nan") for v in mae_list],
            "rmse": [float(v) if np.isfinite(v) else float("nan") for v in rmse_list],
            "n": n_list,
        }
    except Exception:
        return None


def regression_summary(
    *,
    y_true: Any,
    y_pred: Any,
    include_explained_variance: bool = True,
) -> Mapping[str, Any]:
    """Compute a compact set of regression metrics.

    Output fields are stable and JSON-friendly.
    """
    # Fallback-friendly path without numpy: return minimal
    if np is None:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "r2": None,
            "median_abs_error": None,
            "bias": None,
            "pearson_r": None,
            "spearman_r": None,
        }

    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)

    try:
        yt = np.asarray(yt, dtype=float).reshape(-1)
        yp = np.asarray(yp, dtype=float).reshape(-1)
    except Exception:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "r2": None,
            "median_abs_error": None,
            "bias": None,
            "pearson_r": None,
            "spearman_r": None,
        }

    if yt.size == 0 or yp.size == 0 or yt.size != yp.size:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "r2": None,
            "median_abs_error": None,
            "bias": None,
            "pearson_r": None,
            "spearman_r": None,
        }

    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]
    n = int(yt.size)
    if n == 0:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "r2": None,
            "median_abs_error": None,
            "bias": None,
            "pearson_r": None,
            "spearman_r": None,
        }

    residuals = yt - yp
    abs_err = np.abs(residuals)

    # Normalization reference (std of true values) for NRMSE.
    # If std is ~0 (near-constant target), NRMSE is not meaningful.
    try:
        y_std = float(np.std(yt))
        if not np.isfinite(y_std) or y_std <= 0:
            y_std = None
    except Exception:
        y_std = None

    # sklearn metrics when available
    rmse = None
    mae = None
    r2 = None
    med_ae = None
    ev = None
    try:
        if _sk_mse is not None:
            rmse = float(math.sqrt(float(_sk_mse(yt, yp))))
        else:
            rmse = float(np.sqrt(np.mean(residuals * residuals)))
    except Exception:
        rmse = float(np.sqrt(np.mean(residuals * residuals)))

    try:
        if _sk_mae is not None:
            mae = float(_sk_mae(yt, yp))
        else:
            mae = float(np.mean(abs_err))
    except Exception:
        mae = float(np.mean(abs_err))

    try:
        if _sk_r2 is not None:
            r2 = float(_sk_r2(yt, yp))
    except Exception:
        r2 = None

    try:
        if _sk_med_ae is not None:
            med_ae = float(_sk_med_ae(yt, yp))
        else:
            med_ae = float(np.median(abs_err))
    except Exception:
        med_ae = float(np.median(abs_err))

    if include_explained_variance:
        try:
            if _sk_ev is not None:
                ev = float(_sk_ev(yt, yp))
        except Exception:
            ev = None

    pearson_r, spearman_r = _maybe_corr(yt, yp)

    out: Dict[str, Any] = {
        "n": int(n),
        "rmse": _safe_float(rmse) if rmse is not None else None,
        "mae": _safe_float(mae) if mae is not None else None,
        "r2": _safe_float(r2) if r2 is not None else None,
        "median_abs_error": _safe_float(med_ae) if med_ae is not None else None,
        "bias": _safe_float(np.mean(residuals)),
        "pearson_r": _safe_float(pearson_r) if pearson_r is not None else None,
        "spearman_r": _safe_float(spearman_r) if spearman_r is not None else None,
        "residual_std": _safe_float(np.std(residuals)),
        "y_true_min": _safe_float(np.min(yt)),
        "y_true_max": _safe_float(np.max(yt)),
        "y_pred_min": _safe_float(np.min(yp)),
        "y_pred_max": _safe_float(np.max(yp)),
    }

    # Normalized RMSE (NRMSE): RMSE divided by std(y_true), when meaningful.
    try:
        if rmse is not None and y_std is not None and y_std > 0:
            out["nrmse"] = _safe_float(float(rmse) / float(y_std))
        else:
            out["nrmse"] = None
    except Exception:
        out["nrmse"] = None

    if ev is not None:
        out["explained_variance"] = _safe_float(ev)

    return out


def regression_diagnostics(
    *,
    y_true: Any,
    y_pred: Any,
    scatter_max_points: int = 5000,
    residual_hist_bins: int = 40,
    seed: int = 0,
) -> Mapping[str, Any]:
    """Build a full regression diagnostics payload.

    The payload is intentionally compact:
      - summary metrics
      - predicted-vs-true points (downsampled)
      - residual histogram
      - residuals-vs-pred points (downsampled)
      - binned error by true quantiles
    """
    if np is None:
        return {"summary": regression_summary(y_true=y_true, y_pred=y_pred)}

    yt = np.asarray(_as_1d(y_true), dtype=float).reshape(-1)
    yp = np.asarray(_as_1d(y_pred), dtype=float).reshape(-1)
    if yt.size == 0 or yp.size == 0 or yt.size != yp.size:
        return {"summary": regression_summary(y_true=y_true, y_pred=y_pred)}

    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]
    if yt.size == 0:
        return {"summary": regression_summary(y_true=y_true, y_pred=y_pred)}

    residuals = yt - yp

    payload: Dict[str, Any] = {
        "summary": regression_summary(y_true=yt, y_pred=yp),
        "pred_vs_true": downsample_xy(yt, yp, max_points=scatter_max_points, seed=seed),
        "residuals_vs_pred": downsample_xy(yp, residuals, max_points=scatter_max_points, seed=seed),
        "residual_hist": histogram_1d(residuals, n_bins=residual_hist_bins),
        "error_by_true_bin": binned_error_by_true(yt, yp, n_bins=10),
    }

    # Provide the ideal line extent for pred-vs-true (frontend can draw y=x)
    try:
        lo = float(min(float(np.min(yt)), float(np.min(yp))))
        hi = float(max(float(np.max(yt)), float(np.max(yp))))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            payload["ideal_line"] = {"x": [lo, hi], "y": [lo, hi]}
    except Exception:
        pass

    return payload
