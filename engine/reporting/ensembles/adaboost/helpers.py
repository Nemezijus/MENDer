from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from engine.reporting.common.hist import (
    hist_add_inplace as _hist_add_inplace,
    hist_init as _hist_init,
    histogram_minmax_payload as _histogram_minmax_payload,
)


# -----------------------
# hist utilities
# -----------------------
def hist_init(edges: np.ndarray) -> np.ndarray:
    # Keep signature stable for accumulators, delegate implementation.
    return _hist_init(edges, np_mod=np)


def hist_add(counts: np.ndarray, edges: np.ndarray, values: Sequence[float]) -> None:
    _hist_add_inplace(counts=counts, edges=edges, values=values, np_mod=np)


def base_score_hist(scores: Sequence[float], *, bins: int = 20) -> Dict[str, Any]:
    # Keep stable output shape used by the UI.
    vals = [float(x) for x in scores if x is not None]
    return dict(_histogram_minmax_payload(vals, bins=bins, np_mod=np))


# -----------------------
# weight-derived stats
# -----------------------
def effective_n_from_weights(w: Sequence[float]) -> float:
    """Effective number of estimators (ESS) from weights."""
    ww = np.asarray(w, dtype=float)
    ww = ww[ww > 0]
    if ww.size == 0:
        return 0.0
    s1 = float(np.sum(ww))
    s2 = float(np.sum(ww * ww))
    if s2 <= 0:
        return 0.0
    return (s1 * s1) / s2


def weighted_margin_strength_tie(
    preds_row: Sequence[Any],
    weights: Sequence[float],
) -> Tuple[float, float, bool]:
    """Normalized weighted vote margin/strength for one sample (0..1) + tie flag."""
    w = np.asarray(weights, dtype=float)
    m = len(preds_row)
    if w.size != m:
        w = np.ones(m, dtype=float)

    total = float(np.sum(w))
    if total <= 0:
        return 0.0, 0.0, True

    vote_w: Dict[Any, float] = {}
    for p, ww in zip(preds_row, w):
        vote_w[p] = vote_w.get(p, 0.0) + float(ww)

    vals = sorted(vote_w.values(), reverse=True)
    top = float(vals[0]) if vals else 0.0
    second = float(vals[1]) if len(vals) > 1 else 0.0

    strength = top / total
    margin = max(0.0, top - second) / total
    tie = bool(np.isclose(top, second))

    return float(margin), float(strength), tie


# -----------------------
# regression similarity / error helpers
# -----------------------
def corr_matrix_safe(P: np.ndarray) -> np.ndarray:
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


def regression_error_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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


__all__ = [
    "hist_init",
    "hist_add",
    "base_score_hist",
    "effective_n_from_weights",
    "weighted_margin_strength_tie",
    "corr_matrix_safe",
    "mean_absdiff_matrix",
    "regression_error_stats",
    "r2_score_safe",
]