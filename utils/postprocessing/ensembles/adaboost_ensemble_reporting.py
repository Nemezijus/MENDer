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


__all__ = ["AdaBoostEnsembleReportAccumulator"]
