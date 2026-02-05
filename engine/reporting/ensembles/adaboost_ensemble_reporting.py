# utils/postprocessing/ensembles/adaboost_ensemble_reporting.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import _safe_float, _safe_int, _mean_std


def _hist_init(edges: np.ndarray) -> np.ndarray:
    return np.zeros(len(edges) - 1, dtype=float)


def _hist_add(counts: np.ndarray, values: Sequence[float], edges: np.ndarray) -> None:
    h, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    counts += h


def _effective_n_from_weights(w: np.ndarray) -> float:
    """Effective number of estimators (a.k.a. ESS) from weights.
    If weights are uniform -> ESS ~= n. If concentrated -> ESS smaller.
    """
    w = np.asarray(w, dtype=float)
    w = w[w > 0]
    if w.size == 0:
        return 0.0
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if s2 <= 0:
        return 0.0
    return (s1 * s1) / s2


def _weighted_margin_strength_tie(
    preds_row: Sequence[Any],
    weights: np.ndarray,
) -> Tuple[float, float, bool]:
    """Compute normalized weighted vote margin and strength for a single sample.

    - strength: top_weight / total_weight  (0..1)
    - margin: (top_weight - second_weight) / total_weight  (0..1)
    - tie: True if top_weight == second_weight (within tolerance)

    preds_row: predictions from each estimator for the sample
    weights: estimator weights (length m)
    """
    w = np.asarray(weights, dtype=float)
    m = len(preds_row)
    if w.size != m:
        # fallback: treat as uniform if mismatch
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


@dataclass
class AdaBoostEnsembleReportAccumulator:
    metric_name: str
    base_algo: str

    n_estimators: int
    learning_rate: float
    algorithm: Optional[str] = None  # SAMME / SAMME.R (sklearn varies by version)

    _n_folds: int = 0
    _n_eval_total: int = 0

    # vote stats (weighted)
    _margin_sum: float = 0.0
    _strength_sum: float = 0.0
    _tie_count: int = 0
    _margin_hist_edges: np.ndarray = None
    _margin_hist_counts: np.ndarray = None
    _strength_hist_edges: np.ndarray = None
    _strength_hist_counts: np.ndarray = None

    # estimator weights/errors across folds
    _weights_all: List[float] = None
    _errors_all: List[float] = None
    _eff_n_all: List[float] = None

    # ---- diagnostics about fitted stages and weight concentration ----
    _n_estimators_fitted_all: List[int] = None
    _n_nonzero_weights_all: List[int] = None
    _n_nontrivial_weights_all: List[int] = None
    _weight_eps: float = 1e-6
    _weight_mass_topk_all: Dict[int, List[float]] = None

    _weight_hist_edges: np.ndarray = None
    _weight_hist_counts: np.ndarray = None
    _error_hist_edges: np.ndarray = None
    _error_hist_counts: np.ndarray = None

    # optional: base estimator score distribution (if provided by service)
    _base_scores_all: List[float] = None
    _base_score_hist_edges: np.ndarray = None
    _base_score_hist_counts: np.ndarray = None

    @classmethod
    def create(
        cls,
        *,
        metric_name: str,
        base_algo: str,
        n_estimators: int,
        learning_rate: float,
        algorithm: Optional[str] = None,
    ) -> "AdaBoostEnsembleReportAccumulator":
        n = int(n_estimators)

        # Normalized margin/strength are in [0,1]
        edges01 = np.linspace(0.0, 1.0, num=21)

        # Weights can be >1 and vary wildly; use log-ish friendly bins by clipping
        # but keep it simple: 0..maxW with 25 bins; backend can still display.
        # We'll build weights hist dynamically by using a generous range.
        weight_edges = np.linspace(0.0, 5.0, num=26)  # if weights exceed, still counted in last bin by numpy? (no)
        # Better: explicitly clip to last edge before histogram add.
        error_edges = np.linspace(0.0, 1.0, num=21)

        score_edges = np.linspace(0.0, 1.0, num=21)

        return cls(
            metric_name=str(metric_name),
            base_algo=str(base_algo),
            n_estimators=n,
            learning_rate=float(learning_rate),
            algorithm=str(algorithm) if algorithm is not None else None,
            _weights_all=[],
            _errors_all=[],
            _eff_n_all=[],
            _base_scores_all=[],
            _margin_hist_edges=edges01,
            _margin_hist_counts=_hist_init(edges01),
            _strength_hist_edges=edges01,
            _strength_hist_counts=_hist_init(edges01),
            _n_estimators_fitted_all=[],
            _n_nonzero_weights_all=[],
            _n_nontrivial_weights_all=[],
            _weight_eps=1e-6,
            _weight_mass_topk_all={5: [], 10: [], 20: []},
            _weight_hist_edges=weight_edges,
            _weight_hist_counts=_hist_init(weight_edges),
            _error_hist_edges=error_edges,
            _error_hist_counts=_hist_init(error_edges),
            _base_score_hist_edges=score_edges,
            _base_score_hist_counts=_hist_init(score_edges),
        )

    def add_fold(
        self,
        *,
        base_preds: np.ndarray,
        estimator_weights: Sequence[float],
        estimator_errors: Optional[Sequence[float]] = None,
        base_estimator_scores: Optional[Sequence[float]] = None,

        n_estimators_fitted: Optional[int] = None,
        n_nonzero_weights: Optional[int] = None,
        n_nontrivial_weights: Optional[int] = None,
        weight_eps: Optional[float] = None,
        weight_mass_topk: Optional[Dict[int, float]] = None,
    ) -> None:
        """Add fold data.

        base_preds: (n_eval, n_estimators) predictions from each stage estimator (final estimators_)
        estimator_weights: length n_estimators
        estimator_errors: optional length n_estimators (AdaBoost has estimator_errors_)
        base_estimator_scores: optional per-estimator scores on eval set (distribution)
        """
        self._n_folds += 1

        if n_estimators_fitted is not None:
            self._n_estimators_fitted_all.append(int(n_estimators_fitted))
        if n_nonzero_weights is not None:
            self._n_nonzero_weights_all.append(int(n_nonzero_weights))
        if n_nontrivial_weights is not None:
            self._n_nontrivial_weights_all.append(int(n_nontrivial_weights))
        if weight_eps is not None:
            self._weight_eps = float(weight_eps)
        if weight_mass_topk:
            for k, v in weight_mass_topk.items():
                kk = int(k)
                if self._weight_mass_topk_all is None:
                    self._weight_mass_topk_all = {}
                self._weight_mass_topk_all.setdefault(kk, []).append(float(v))

        P = np.asarray(base_preds)
        if P.ndim != 2:
            return
        n_eval, m = int(P.shape[0]), int(P.shape[1])
        if n_eval <= 0 or m <= 0:
            return

        w = np.asarray(estimator_weights, dtype=float)
        if w.size != m:
            # best-effort align: truncate/pad with ones
            if w.size > m:
                w = w[:m]
            else:
                w = np.concatenate([w, np.ones(m - w.size, dtype=float)])

        self._n_eval_total += n_eval

        # weighted vote stats
        margins = np.zeros(n_eval, dtype=float)
        strengths = np.zeros(n_eval, dtype=float)
        ties = 0
        for i in range(n_eval):
            margin, strength, tie = _weighted_margin_strength_tie(P[i, :], w)
            margins[i] = margin
            strengths[i] = strength
            ties += 1 if tie else 0

        self._tie_count += ties
        self._margin_sum += float(np.sum(margins))
        self._strength_sum += float(np.sum(strengths))
        _hist_add(self._margin_hist_counts, margins, self._margin_hist_edges)
        _hist_add(self._strength_hist_counts, strengths, self._strength_hist_edges)

        # weights/errors stats
        w_pos = w[w > 0]
        if w_pos.size:
            self._weights_all.extend([float(x) for x in w_pos.tolist()])

            # clip to hist range upper edge - tiny epsilon, so we don't drop out-of-range
            w_clip = np.clip(w_pos, self._weight_hist_edges[0], self._weight_hist_edges[-1] - 1e-12)
            _hist_add(self._weight_hist_counts, w_clip, self._weight_hist_edges)

            self._eff_n_all.append(float(_effective_n_from_weights(w_pos)))

        if estimator_errors is not None:
            e = np.asarray(estimator_errors, dtype=float)
            e = e[np.isfinite(e)]
            if e.size:
                self._errors_all.extend([float(x) for x in e.tolist()])
                e_clip = np.clip(e, self._error_hist_edges[0], self._error_hist_edges[-1] - 1e-12)
                _hist_add(self._error_hist_counts, e_clip, self._error_hist_edges)

        # optional: base estimator score distribution
        if base_estimator_scores is not None:
            vals = [float(s) for s in base_estimator_scores if s is not None]
            if vals:
                self._base_scores_all.extend(vals)
                s_clip = np.clip(np.asarray(vals, dtype=float), 0.0, 1.0 - 1e-12)
                _hist_add(self._base_score_hist_counts, s_clip, self._base_score_hist_edges)

    def finalize(self) -> Dict[str, Any]:
        if self._n_eval_total <= 0:
            mean_margin = None
            mean_strength = None
            tie_rate = None
            margin_hist = None
            strength_hist = None
        else:
            denom = float(self._n_eval_total)
            mean_margin = float(self._margin_sum / denom)
            mean_strength = float(self._strength_sum / denom)
            tie_rate = float(self._tie_count / denom)
            margin_hist = {
                "edges": [float(x) for x in self._margin_hist_edges.tolist()],
                "counts": [float(x) for x in self._margin_hist_counts.tolist()],
            }
            strength_hist = {
                "edges": [float(x) for x in self._strength_hist_edges.tolist()],
                "counts": [float(x) for x in self._strength_hist_counts.tolist()],
            }

        w_mean, w_std = _mean_std(self._weights_all or [])
        e_mean, e_std = _mean_std(self._errors_all or [])
        eff_mean, eff_std = _mean_std(self._eff_n_all or [])

        bs_mean, bs_std = _mean_std(self._base_scores_all or [])

        fitted_mean, fitted_std = _mean_std(self._n_estimators_fitted_all or [])
        nz_mean, nz_std = _mean_std(self._n_nonzero_weights_all or [])
        nt_mean, nt_std = _mean_std(self._n_nontrivial_weights_all or [])

        topk_mean: Dict[int, float] = {}
        topk_std: Dict[int, float] = {}
        if self._weight_mass_topk_all:
            for k, vals in self._weight_mass_topk_all.items():
                m_k, s_k = _mean_std(vals or [])
                if vals:
                    topk_mean[int(k)] = float(m_k)
                    topk_std[int(k)] = float(s_k)

        return {
            "kind": "adaboost",
            "task": "classification",
            "metric_name": self.metric_name,
            "adaboost": {
                "base_algo": self.base_algo,
                "n_estimators": _safe_int(self.n_estimators),
                "learning_rate": _safe_float(self.learning_rate),
                "algorithm": self.algorithm,
            },
            "vote": {
                "mean_margin": _safe_float(mean_margin) if mean_margin is not None else None,
                "mean_strength": _safe_float(mean_strength) if mean_strength is not None else None,
                "tie_rate": _safe_float(tie_rate) if tie_rate is not None else None,
                "margin_hist": margin_hist,
                "strength_hist": strength_hist,
            },
            "weights": {
                "mean": _safe_float(w_mean) if self._weights_all else None,
                "std": _safe_float(w_std) if self._weights_all else None,
                "effective_n_mean": _safe_float(eff_mean) if self._eff_n_all else None,
                "effective_n_std": _safe_float(eff_std) if self._eff_n_all else None,
                "hist": {
                    "edges": [float(x) for x in self._weight_hist_edges.tolist()],
                    "counts": [float(x) for x in self._weight_hist_counts.tolist()],
                }
                if self._weights_all
                else None,
            },
            "stages": {
                "n_estimators_configured": _safe_int(self.n_estimators),
                "n_estimators_fitted_mean": _safe_float(fitted_mean) if self._n_estimators_fitted_all else None,
                "n_estimators_fitted_std": _safe_float(fitted_std) if self._n_estimators_fitted_all else None,
                "n_nonzero_weights_mean": _safe_float(nz_mean) if self._n_nonzero_weights_all else None,
                "n_nonzero_weights_std": _safe_float(nz_std) if self._n_nonzero_weights_all else None,
                "n_nontrivial_weights_mean": _safe_float(nt_mean) if self._n_nontrivial_weights_all else None,
                "n_nontrivial_weights_std": _safe_float(nt_std) if self._n_nontrivial_weights_all else None,
                "weight_eps": _safe_float(self._weight_eps),
                "weight_mass_topk_mean": {str(k): _safe_float(v) for k, v in (topk_mean or {}).items()} or None,
                "weight_mass_topk_std": {str(k): _safe_float(v) for k, v in (topk_std or {}).items()} or None,
            },
            "errors": {
                "mean": _safe_float(e_mean) if self._errors_all else None,
                "std": _safe_float(e_std) if self._errors_all else None,
                "hist": {
                    "edges": [float(x) for x in self._error_hist_edges.tolist()],
                    "counts": [float(x) for x in self._error_hist_counts.tolist()],
                }
                if self._errors_all
                else None,
            },
            "base_estimator_scores": {
                "mean": _safe_float(bs_mean) if self._base_scores_all else None,
                "std": _safe_float(bs_std) if self._base_scores_all else None,
                "hist": {
                    "edges": [float(x) for x in self._base_score_hist_edges.tolist()],
                    "counts": [float(x) for x in self._base_score_hist_counts.tolist()],
                }
                if self._base_scores_all
                else None,
            },
            "meta": {
                "n_folds": _safe_int(self._n_folds),
                "n_eval_samples_total": _safe_int(self._n_eval_total),
            },
        }



def _regression_error_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


def _corr_matrix_safe(P: np.ndarray) -> np.ndarray:
    """Correlation matrix across estimator predictions with NaN handling."""
    try:
        C = np.corrcoef(P, rowvar=False)
        C = np.asarray(C, dtype=float)
        if C.ndim != 2:
            return np.zeros((P.shape[1], P.shape[1]), dtype=float)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(C, 1.0)
        return C
    except Exception:
        m = int(P.shape[1]) if P.ndim == 2 else 0
        return np.eye(m, dtype=float)


def _mean_absdiff_matrix(P: np.ndarray) -> np.ndarray:
    """Mean absolute difference matrix across estimator predictions."""
    P = np.asarray(P, dtype=float)
    n, m = P.shape
    out = np.zeros((m, m), dtype=float)
    if n <= 0 or m <= 0:
        return out
    for i in range(m):
        xi = P[:, i]
        for j in range(i + 1, m):
            v = float(np.mean(np.abs(xi - P[:, j])))
            out[i, j] = v
            out[j, i] = v
    return out



def _base_score_hist(scores: Sequence[float], *, bins: int = 20) -> Dict[str, Any]:
    """Compute a histogram for pooled stage/base-estimator scores.

    Edges are data-driven (min..max). If all values are equal, a tiny range is added.
    """
    vals = np.asarray([float(x) for x in scores if x is not None], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"edges": [], "counts": []}

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if np.isclose(vmin, vmax):
        eps = 1e-6 if vmax == 0.0 else abs(vmax) * 1e-6
        vmin -= eps
        vmax += eps

    edges = np.linspace(vmin, vmax, num=int(bins) + 1, dtype=float)
    counts, edges = np.histogram(vals, bins=edges)
    return {"edges": [float(x) for x in edges.tolist()], "counts": [float(x) for x in counts.tolist()]}


@dataclass
class AdaBoostEnsembleRegressorReportAccumulator:
    """Accumulate regression-specific AdaBoost insights across folds."""

    metric_name: str
    base_algo: str

    n_estimators: int
    learning_rate: float
    loss: Optional[str] = None  # linear / square / exponential (sklearn)

    _n_folds: int = 0
    _n_eval_total: int = 0

    # weights/errors across folds
    _weights_all: List[float] = None
    _errors_all: List[float] = None
    _eff_n_all: List[float] = None

    # diagnostics
    _n_estimators_fitted_all: List[int] = None
    _n_nonzero_weights_all: List[int] = None
    _n_nontrivial_weights_all: List[int] = None
    _weight_eps: float = 1e-6
    _weight_mass_topk_all: Dict[int, List[float]] = None

    _weight_hist_edges: np.ndarray = None
    _weight_hist_counts: np.ndarray = None
    _error_hist_edges: np.ndarray = None
    _error_hist_counts: np.ndarray = None

    # diversity/similarity matrices (weighted by n_eval)
    _corr_sum: np.ndarray = None
    _absdiff_sum: np.ndarray = None
    _pred_spread_sum: float = 0.0

    # errors (weighted by n_eval)
    _ens_rmse_sum: float = 0.0
    _ens_mae_sum: float = 0.0
    _ens_med_sum: float = 0.0

    _best_rmse_sum: float = 0.0
    _best_mae_sum: float = 0.0
    _best_med_sum: float = 0.0

    _gain_rmse_sum: float = 0.0
    _gain_mae_sum: float = 0.0
    _gain_med_sum: float = 0.0

    # optional: base estimator score distribution
    _base_scores_all: List[float] = None

    @classmethod
    def create(
        cls,
        *,
        metric_name: str,
        base_algo: str,
        n_estimators: int,
        learning_rate: float,
        loss: Optional[str] = None,
    ) -> "AdaBoostEnsembleRegressorReportAccumulator":
        n = int(n_estimators)
        weight_edges = np.linspace(0.0, 5.0, num=26)
        error_edges = np.linspace(0.0, 1.0, num=21)

        return cls(
            metric_name=str(metric_name),
            base_algo=str(base_algo),
            n_estimators=n,
            learning_rate=float(learning_rate),
            loss=str(loss) if loss is not None else None,
            _weights_all=[],
            _errors_all=[],
            _eff_n_all=[],
            _n_estimators_fitted_all=[],
            _n_nonzero_weights_all=[],
            _n_nontrivial_weights_all=[],
            _weight_eps=1e-6,
            _weight_mass_topk_all={5: [], 10: [], 20: []},
            _weight_hist_edges=weight_edges,
            _weight_hist_counts=np.zeros(len(weight_edges) - 1, dtype=float),
            _error_hist_edges=error_edges,
            _error_hist_counts=np.zeros(len(error_edges) - 1, dtype=float),
            _corr_sum=np.zeros((n, n), dtype=float) if n > 0 else np.zeros((0, 0), dtype=float),
            _absdiff_sum=np.zeros((n, n), dtype=float) if n > 0 else np.zeros((0, 0), dtype=float),
            _base_scores_all=[],
        )

    def add_fold(
        self,
        *,
        y_true: np.ndarray,
        ensemble_pred: np.ndarray,
        base_preds: np.ndarray,
        estimator_weights: Sequence[float],
        estimator_errors: Optional[Sequence[float]] = None,
        base_estimator_scores: Optional[Sequence[float]] = None,
        n_estimators_fitted: Optional[int] = None,
        n_nonzero_weights: Optional[int] = None,
        n_nontrivial_weights: Optional[int] = None,
        weight_eps: Optional[float] = None,
        weight_mass_topk: Optional[Dict[int, float]] = None,
    ) -> None:
        self._n_folds += 1

        if n_estimators_fitted is not None:
            self._n_estimators_fitted_all.append(int(n_estimators_fitted))
        if n_nonzero_weights is not None:
            self._n_nonzero_weights_all.append(int(n_nonzero_weights))
        if n_nontrivial_weights is not None:
            self._n_nontrivial_weights_all.append(int(n_nontrivial_weights))
        if weight_eps is not None:
            self._weight_eps = float(weight_eps)
        if weight_mass_topk:
            for k, v in weight_mass_topk.items():
                kk = int(k)
                self._weight_mass_topk_all.setdefault(kk, []).append(float(v))

        P = np.asarray(base_preds, dtype=float)
        if P.ndim != 2:
            return
        n_eval, m = int(P.shape[0]), int(P.shape[1])
        if n_eval <= 0 or m <= 0:
            return

        w = np.asarray(estimator_weights, dtype=float)
        if w.size != m:
            if w.size > m:
                w = w[:m]
            else:
                w = np.concatenate([w, np.ones(m - w.size, dtype=float)])

        self._n_eval_total += n_eval

        # similarity/diversity
        spread = np.std(P, axis=1)
        self._pred_spread_sum += float(np.sum(spread))

        C = _corr_matrix_safe(P)
        D = _mean_absdiff_matrix(P)
        if self._corr_sum.shape == (m, m):
            self._corr_sum += C * float(n_eval)
        if self._absdiff_sum.shape == (m, m):
            self._absdiff_sum += D * float(n_eval)

        # weights/errors stats
        w_pos = w[w > 0]
        if w_pos.size:
            self._weights_all.extend([float(x) for x in w_pos.tolist()])
            w_clip = np.clip(w_pos, self._weight_hist_edges[0], self._weight_hist_edges[-1] - 1e-12)
            h, _ = np.histogram(w_clip, bins=self._weight_hist_edges)
            self._weight_hist_counts += h
            # effective n
            s1 = float(np.sum(w_pos))
            s2 = float(np.sum(w_pos * w_pos))
            eff = (s1 * s1) / s2 if s2 > 0 else 0.0
            self._eff_n_all.append(float(eff))

        if estimator_errors is not None:
            e = np.asarray(estimator_errors, dtype=float)
            e = e[np.isfinite(e)]
            if e.size:
                self._errors_all.extend([float(x) for x in e.tolist()])
                e_clip = np.clip(e, self._error_hist_edges[0], self._error_hist_edges[-1] - 1e-12)
                h, _ = np.histogram(e_clip, bins=self._error_hist_edges)
                self._error_hist_counts += h

        # ensemble vs best weak learner
        ens_err = _regression_error_stats(y_true, ensemble_pred)
        self._ens_rmse_sum += ens_err["rmse"] * float(n_eval)
        self._ens_mae_sum += ens_err["mae"] * float(n_eval)
        self._ens_med_sum += ens_err["median_ae"] * float(n_eval)

        yt = np.asarray(y_true, dtype=float).ravel()
        best_rmse = None
        best_mae = None
        best_med = None
        for j in range(m):
            e = _regression_error_stats(yt, P[:, j])
            if best_rmse is None or e["rmse"] < best_rmse:
                best_rmse, best_mae, best_med = e["rmse"], e["mae"], e["median_ae"]

        if best_rmse is None:
            best_rmse = best_mae = best_med = 0.0

        self._best_rmse_sum += float(best_rmse) * float(n_eval)
        self._best_mae_sum += float(best_mae) * float(n_eval)
        self._best_med_sum += float(best_med) * float(n_eval)

        self._gain_rmse_sum += float(best_rmse - ens_err["rmse"]) * float(n_eval)
        self._gain_mae_sum += float(best_mae - ens_err["mae"]) * float(n_eval)
        self._gain_med_sum += float(best_med - ens_err["median_ae"]) * float(n_eval)

        if base_estimator_scores is not None:
            vals = [float(s) for s in base_estimator_scores if s is not None]
            if vals:
                self._base_scores_all.extend(vals)

    def finalize(self) -> Dict[str, Any]:
        denom = float(self._n_eval_total) if self._n_eval_total > 0 else 1.0

        # matrices
        MAX_MATRIX_ESTIMATORS = 80
        corr_out = None
        absdiff_out = None
        labels_out = None
        pairwise_mean_corr = None
        pairwise_mean_absdiff = None

        if self.n_estimators > 0 and self._corr_sum.size:
            C = self._corr_sum / denom
            D = self._absdiff_sum / denom
            m = C.shape[0]
            if m > 1:
                mask = ~np.eye(m, dtype=bool)
                pairwise_mean_corr = float(np.mean(C[mask]))
                pairwise_mean_absdiff = float(np.mean(D[mask]))
            else:
                pairwise_mean_corr = 1.0
                pairwise_mean_absdiff = 0.0

            if self.n_estimators <= MAX_MATRIX_ESTIMATORS:
                corr_out = C.tolist()
                absdiff_out = D.tolist()
                labels_out = [f"est_{i+1}" for i in range(self.n_estimators)]

        pred_spread_mean = float(self._pred_spread_sum / denom) if denom else 0.0

        # weights/errors summaries
        w_mean = float(np.mean(self._weights_all)) if self._weights_all else 0.0
        w_std = float(np.std(self._weights_all)) if self._weights_all else 0.0
        e_mean = float(np.mean(self._errors_all)) if self._errors_all else 0.0
        e_std = float(np.std(self._errors_all)) if self._errors_all else 0.0
        eff_mean = float(np.mean(self._eff_n_all)) if self._eff_n_all else 0.0
        eff_std = float(np.std(self._eff_n_all)) if self._eff_n_all else 0.0

        # stage diagnostics
        fitted_mean = float(np.mean(self._n_estimators_fitted_all)) if self._n_estimators_fitted_all else 0.0
        fitted_std = float(np.std(self._n_estimators_fitted_all)) if self._n_estimators_fitted_all else 0.0
        nz_mean = float(np.mean(self._n_nonzero_weights_all)) if self._n_nonzero_weights_all else 0.0
        nz_std = float(np.std(self._n_nonzero_weights_all)) if self._n_nonzero_weights_all else 0.0
        nt_mean = float(np.mean(self._n_nontrivial_weights_all)) if self._n_nontrivial_weights_all else 0.0
        nt_std = float(np.std(self._n_nontrivial_weights_all)) if self._n_nontrivial_weights_all else 0.0

        topk_mean = {}
        topk_std = {}
        if self._weight_mass_topk_all:
            for k, vals in self._weight_mass_topk_all.items():
                if not vals:
                    continue
                topk_mean[int(k)] = float(np.mean(vals))
                topk_std[int(k)] = float(np.std(vals))

        # errors (weighted)
        ens_rmse = float(self._ens_rmse_sum / denom)
        ens_mae = float(self._ens_mae_sum / denom)
        ens_med = float(self._ens_med_sum / denom)

        best_rmse = float(self._best_rmse_sum / denom)
        best_mae = float(self._best_mae_sum / denom)
        best_med = float(self._best_med_sum / denom)

        gain_rmse = float(self._gain_rmse_sum / denom)
        gain_mae = float(self._gain_mae_sum / denom)
        gain_med = float(self._gain_med_sum / denom)

        bs_mean = float(np.mean(self._base_scores_all)) if self._base_scores_all else 0.0
        bs_std = float(np.std(self._base_scores_all)) if self._base_scores_all else 0.0

        return {
            "kind": "adaboost",
            "task": "regression",
            "metric_name": self.metric_name,
            "adaboost": {
                "base_algo": self.base_algo,
                "n_estimators": _safe_int(self.n_estimators),
                "learning_rate": _safe_float(self.learning_rate),
                "loss": self.loss,
            },
            "diversity": {
                "pairwise_mean_corr": _safe_float(pairwise_mean_corr) if pairwise_mean_corr is not None else None,
                "pairwise_mean_absdiff": _safe_float(pairwise_mean_absdiff) if pairwise_mean_absdiff is not None else None,
                "prediction_spread_mean": _safe_float(pred_spread_mean),
                "labels": labels_out,
                "pairwise_corr": corr_out,
                "pairwise_absdiff": absdiff_out,
            },
            "weights": {
                "mean": _safe_float(w_mean) if self._weights_all else None,
                "std": _safe_float(w_std) if self._weights_all else None,
                "effective_n_mean": _safe_float(eff_mean) if self._eff_n_all else None,
                "effective_n_std": _safe_float(eff_std) if self._eff_n_all else None,
                "hist": {
                    "edges": [float(x) for x in self._weight_hist_edges.tolist()],
                    "counts": [float(x) for x in self._weight_hist_counts.tolist()],
                } if self._weights_all else None,
            },
            "stages": {
                "n_estimators_configured": _safe_int(self.n_estimators),
                "n_estimators_fitted_mean": _safe_float(fitted_mean) if self._n_estimators_fitted_all else None,
                "n_estimators_fitted_std": _safe_float(fitted_std) if self._n_estimators_fitted_all else None,
                "n_nonzero_weights_mean": _safe_float(nz_mean) if self._n_nonzero_weights_all else None,
                "n_nonzero_weights_std": _safe_float(nz_std) if self._n_nonzero_weights_all else None,
                "n_nontrivial_weights_mean": _safe_float(nt_mean) if self._n_nontrivial_weights_all else None,
                "n_nontrivial_weights_std": _safe_float(nt_std) if self._n_nontrivial_weights_all else None,
                "weight_eps": _safe_float(self._weight_eps),
                "weight_mass_topk_mean": {str(k): _safe_float(v) for k, v in (topk_mean or {}).items()} or None,
                "weight_mass_topk_std": {str(k): _safe_float(v) for k, v in (topk_std or {}).items()} or None,
            },
            "errors": {
                "mean": _safe_float(e_mean) if self._errors_all else None,
                "std": _safe_float(e_std) if self._errors_all else None,
                "hist": {
                    "edges": [float(x) for x in self._error_hist_edges.tolist()],
                    "counts": [float(x) for x in self._error_hist_counts.tolist()],
                } if self._errors_all else None,
            },
            "model_errors": {
                "ensemble": {"rmse": _safe_float(ens_rmse), "mae": _safe_float(ens_mae), "median_ae": _safe_float(ens_med)},
                "best_base": {"rmse": _safe_float(best_rmse), "mae": _safe_float(best_mae), "median_ae": _safe_float(best_med)},
                "gain_vs_best": {"rmse_reduction": _safe_float(gain_rmse), "mae_reduction": _safe_float(gain_mae), "median_ae_reduction": _safe_float(gain_med), "r2": None},
                "n_total": _safe_int(self._n_eval_total),
            },
            "base_estimator_scores": {
                "mean": _safe_float(bs_mean) if self._base_scores_all else None,
                "std": _safe_float(bs_std) if self._base_scores_all else None,
                "hist": _base_score_hist(self._base_scores_all) if self._base_scores_all else None,
            },
            "meta": {
                "n_folds": _safe_int(self._n_folds),
                "n_eval_samples_total": _safe_int(self._n_eval_total),
            },
        }


__all__ = ["AdaBoostEnsembleReportAccumulator", "AdaBoostEnsembleRegressorReportAccumulator"]