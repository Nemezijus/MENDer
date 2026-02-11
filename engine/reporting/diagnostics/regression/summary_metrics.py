from __future__ import annotations

from typing import Any, Dict, Mapping

import math

from .common import as_1d, maybe_corr, numpy, safe_float

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


def regression_summary(
    *,
    y_true: Any,
    y_pred: Any,
    include_explained_variance: bool = True,
) -> Mapping[str, Any]:
    """Compute a compact set of regression metrics.

    Output fields are stable and JSON-friendly.
    """
    np = numpy()

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

    yt = as_1d(y_true)
    yp = as_1d(y_pred)

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
    try:
        y_std = float(np.std(yt))
        if not np.isfinite(y_std) or y_std <= 0:
            y_std = None
    except Exception:
        y_std = None

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

    pearson_r, spearman_r = maybe_corr(yt, yp)

    out: Dict[str, Any] = {
        "n": int(n),
        "rmse": safe_float(rmse) if rmse is not None else None,
        "mae": safe_float(mae) if mae is not None else None,
        "r2": safe_float(r2) if r2 is not None else None,
        "median_abs_error": safe_float(med_ae) if med_ae is not None else None,
        "bias": safe_float(np.mean(residuals)),
        "pearson_r": safe_float(pearson_r) if pearson_r is not None else None,
        "spearman_r": safe_float(spearman_r) if spearman_r is not None else None,
        "residual_std": safe_float(np.std(residuals)),
        "y_true_min": safe_float(np.min(yt)),
        "y_true_max": safe_float(np.max(yt)),
        "y_pred_min": safe_float(np.min(yp)),
        "y_pred_max": safe_float(np.max(yp)),
    }

    try:
        if rmse is not None and y_std is not None and y_std > 0:
            out["nrmse"] = safe_float(float(rmse) / float(y_std))
        else:
            out["nrmse"] = None
    except Exception:
        out["nrmse"] = None

    if ev is not None:
        out["explained_variance"] = safe_float(ev)

    return out
