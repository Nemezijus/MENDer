from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from engine.reporting.common.hist import histogram_minmax_payload as _histogram_minmax_payload

from engine.reporting.ensembles.common import vote_margin_and_strength


def oob_coverage_from_decision_function(oob_decision: Any) -> Optional[float]:
    """Estimate OOB coverage from sklearn-style oob_decision_function_.

    Typically shape: (n_train, n_classes) with NaNs for samples that never had OOB preds.
    Coverage = fraction of rows that have no NaNs.
    """
    try:
        a = np.asarray(oob_decision)
        if a.ndim < 2:
            return None
        ok = ~np.any(np.isnan(a), axis=1)
        if ok.size == 0:
            return None
        return float(np.mean(ok))
    except Exception:
        return None


def oob_coverage_from_prediction(oob_pred: Any) -> Optional[float]:
    """Estimate OOB coverage from sklearn-style oob_prediction_ (regression).

    oob_prediction_ is typically shape: (n_train,) with NaNs for samples that never had OOB preds.
    Coverage = fraction of entries that are finite.
    """
    try:
        a = np.asarray(oob_pred, dtype=float).ravel()
        if a.size == 0:
            return None
        ok = np.isfinite(a)
        if ok.size == 0:
            return None
        return float(np.mean(ok))
    except Exception:
        return None


def corr_matrix_safe(P: np.ndarray) -> np.ndarray:
    """Correlation matrix across estimator predictions with NaN handling."""
    try:
        C = np.corrcoef(np.asarray(P, dtype=float), rowvar=False)
        C = np.asarray(C, dtype=float)
        if C.ndim != 2:
            return np.zeros((P.shape[1], P.shape[1]), dtype=float)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(C, 1.0)
        return C
    except Exception:
        return np.zeros((P.shape[1], P.shape[1]), dtype=float)


def mean_absdiff_matrix(P: np.ndarray) -> np.ndarray:
    """Mean absolute difference matrix across estimator predictions."""
    X = np.asarray(P, dtype=float)
    if X.ndim != 2:
        return np.zeros((0, 0), dtype=float)
    m = X.shape[1]
    out = np.zeros((m, m), dtype=float)
    if m == 0:
        return out
    for i in range(m):
        for j in range(i, m):
            d = np.abs(X[:, i] - X[:, j])
            v = float(np.mean(d)) if d.size else 0.0
            out[i, j] = v
            out[j, i] = v
    return out


def median_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.size == 0 or yp.size == 0 or yt.size != yp.size:
        return 0.0
    return float(np.median(np.abs(yt - yp)))


def regression_error_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute simple regression error stats without external deps."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.size == 0 or yp.size == 0 or yt.size != yp.size:
        return {"rmse": 0.0, "mae": 0.0, "median_ae": 0.0}
    err = yt - yp
    rmse = float(np.sqrt(np.mean(err * err)))
    ae = np.abs(err)
    mae = float(np.mean(ae))
    med = float(np.median(ae))
    return {"rmse": rmse, "mae": mae, "median_ae": med}


def r2_score_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.size == 0 or yp.size == 0 or yt.size != yp.size:
        return 0.0
    ss_res = float(np.sum((yt - yp) ** 2))
    y_mean = float(np.mean(yt))
    ss_tot = float(np.sum((yt - y_mean) ** 2))
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def base_score_hist(scores: Sequence[float], *, bins: int = 20) -> Dict[str, Any]:
    """Histogram for pooled base-estimator scores (data-driven min..max)."""
    vals = [float(x) for x in scores if x is not None]
    return dict(_histogram_minmax_payload(vals, bins=bins, np_mod=np))


__all__ = [
    "vote_margin_and_strength",
    "oob_coverage_from_decision_function",
    "oob_coverage_from_prediction",
    "corr_matrix_safe",
    "mean_absdiff_matrix",
    "regression_error_stats",
    "r2_score_safe",
    "base_score_hist",
]