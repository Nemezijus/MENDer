from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from engine.reporting.ensembles.accumulators import FoldAccumulatorBase
from engine.reporting.ensembles.common import _mean_std, _safe_float, _safe_int

from .helpers import (
    base_score_hist,
    effective_n_from_weights,
    hist_add,
    hist_init,
    weighted_margin_strength_tie,
)


@dataclass
class AdaBoostEnsembleReportAccumulator(FoldAccumulatorBase):
    """Accumulate AdaBoost classifier diagnostics across folds.

    This restores the legacy report schema expected by the current frontend.
    """

    metric_name: str
    base_algo: str
    n_estimators: int
    learning_rate: float
    algorithm: Optional[str] = None

    _n_eval_total: int = 0

    _margin_sum: float = 0.0
    _strength_sum: float = 0.0
    _tie_count: int = 0

    _margin_hist_edges: np.ndarray = None
    _margin_hist_counts: np.ndarray = None
    _strength_hist_edges: np.ndarray = None
    _strength_hist_counts: np.ndarray = None

    _weight_hist_edges: np.ndarray = None
    _weight_hist_counts: np.ndarray = None

    _error_hist_edges: np.ndarray = None
    _error_hist_counts: np.ndarray = None

    _base_score_hist_edges: np.ndarray = None
    _base_score_hist_counts: np.ndarray = None

    _weights_all: List[float] = None
    _errors_all: List[float] = None
    _eff_n_all: List[float] = None
    _base_scores_all: List[float] = None

    _n_estimators_fitted_all: List[int] = None
    _n_nonzero_weights_all: List[int] = None
    _n_nontrivial_weights_all: List[int] = None
    _weight_eps: float = 1e-12
    _weight_mass_topk_all: Optional[Dict[int, List[float]]] = None

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
        m = int(n_estimators)

        edges01 = np.linspace(0.0, 1.0, num=21, dtype=float)
        weight_edges = np.linspace(0.0, 5.0, num=26, dtype=float)  # clip > 5
        error_edges = np.linspace(0.0, 1.0, num=21, dtype=float)
        base_score_edges = np.linspace(0.0, 1.0, num=21, dtype=float)

        return cls(
            metric_name=str(metric_name),
            base_algo=str(base_algo),
            n_estimators=m,
            learning_rate=float(learning_rate),
            algorithm=algorithm,
            _margin_hist_edges=edges01,
            _margin_hist_counts=hist_init(edges01),
            _strength_hist_edges=edges01,
            _strength_hist_counts=hist_init(edges01),
            _weight_hist_edges=weight_edges,
            _weight_hist_counts=hist_init(weight_edges),
            _error_hist_edges=error_edges,
            _error_hist_counts=hist_init(error_edges),
            _base_score_hist_edges=base_score_edges,
            _base_score_hist_counts=hist_init(base_score_edges),
            _weights_all=[],
            _errors_all=[],
            _eff_n_all=[],
            _base_scores_all=[],
            _n_estimators_fitted_all=[],
            _n_nonzero_weights_all=[],
            _n_nontrivial_weights_all=[],
            _weight_mass_topk_all={},
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
        **_: Any,
    ) -> None:
        """Add one fold.

        Backend historically provides extra diagnostic kwargs; we accept and ignore
        unknown keys via **_.
        """
        self._bump_fold()

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
            margin, strength, tie = weighted_margin_strength_tie(P[i, :], w)
            margins[i] = margin
            strengths[i] = strength
            ties += 1 if tie else 0

        self._tie_count += int(ties)
        self._margin_sum += float(np.sum(margins))
        self._strength_sum += float(np.sum(strengths))
        hist_add(self._margin_hist_counts, self._margin_hist_edges, margins)
        hist_add(self._strength_hist_counts, self._strength_hist_edges, strengths)

        # weights/errors stats
        w_pos = w[w > 0]
        if w_pos.size:
            self._weights_all.extend([float(x) for x in w_pos.tolist()])

            w_clip = np.clip(w_pos, self._weight_hist_edges[0], self._weight_hist_edges[-1] - 1e-12)
            hist_add(self._weight_hist_counts, self._weight_hist_edges, w_clip)

            self._eff_n_all.append(float(effective_n_from_weights(w_pos)))

        if estimator_errors is not None:
            e = np.asarray(estimator_errors, dtype=float)
            e = e[np.isfinite(e)]
            if e.size:
                self._errors_all.extend([float(x) for x in e.tolist()])
                e_clip = np.clip(e, self._error_hist_edges[0], self._error_hist_edges[-1] - 1e-12)
                hist_add(self._error_hist_counts, self._error_hist_edges, e_clip)

        if base_estimator_scores is not None:
            vals = [float(s) for s in base_estimator_scores if s is not None]
            if vals:
                self._base_scores_all.extend(vals)
                s_clip = np.clip(np.asarray(vals, dtype=float), 0.0, 1.0 - 1e-12)
                hist_add(self._base_score_hist_counts, self._base_score_hist_edges, s_clip)

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