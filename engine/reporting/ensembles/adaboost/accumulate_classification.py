from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..common import _mean_std, _safe_float, _safe_int
from .helpers import (
    _effective_n_from_weights,
    _hist_add,
    _hist_init,
    _weighted_margin_strength_tie,
)

from engine.reporting.ensembles.accumulators import FoldAccumulatorBase


@dataclass
class AdaBoostEnsembleReportAccumulator(FoldAccumulatorBase):
    metric_name: str
    base_algo: str

    n_estimators: int
    learning_rate: float
    algorithm: Optional[str] = None  # SAMME / SAMME.R (sklearn varies by version)

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
        edges_margin, edges_strength = _hist_init(n_estimators=m)
        return cls(
            metric_name=str(metric_name),
            base_algo=str(base_algo),
            n_estimators=m,
            learning_rate=float(learning_rate),
            algorithm=str(algorithm) if algorithm is not None else None,
            _margin_hist_edges=edges_margin,
            _margin_hist_counts=np.zeros(len(edges_margin) - 1, dtype=float),
            _strength_hist_edges=edges_strength,
            _strength_hist_counts=np.zeros(len(edges_strength) - 1, dtype=float),
            _weights_all=[],
            _errors_all=[],
        )

    def add_fold(
        self,
        *,
        base_preds: np.ndarray,
        estimator_weights: Optional[Sequence[float]] = None,
        estimator_errors: Optional[Sequence[float]] = None,
    ) -> None:
        self._bump_fold()

        P = np.asarray(base_preds)
        if P.ndim != 2:
            return

        n = int(P.shape[0])
        self._n_eval_total += n

        w = np.asarray(estimator_weights, dtype=float).ravel() if estimator_weights is not None else None
        margins = np.zeros(n, dtype=float)
        strengths = np.zeros(n, dtype=float)
        ties = 0
        for r in range(n):
            margin, strength, tie = _weighted_margin_strength_tie(P[r, :], weights=w)
            margins[r] = float(margin)
            strengths[r] = float(strength)
            ties += 1 if tie else 0

        self._tie_count += int(ties)
        self._margin_sum += float(np.sum(margins))
        self._strength_sum += float(np.sum(strengths))

        _hist_add(self._margin_hist_counts, self._margin_hist_edges, margins)
        _hist_add(self._strength_hist_counts, self._strength_hist_edges, strengths)

        if estimator_weights is not None:
            for x in list(estimator_weights):
                try:
                    self._weights_all.append(float(x))
                except Exception:
                    pass

        if estimator_errors is not None:
            for x in list(estimator_errors):
                try:
                    self._errors_all.append(float(x))
                except Exception:
                    pass

    def finalize(self) -> Dict[str, Any]:
        n_total = float(self._n_eval_total) if self._n_eval_total > 0 else 1.0
        mean_margin = float(self._margin_sum / n_total)
        mean_strength = float(self._strength_sum / n_total)
        tie_rate = float(self._tie_count / n_total)

        w_mean, w_std = _mean_std(self._weights_all or [])
        e_mean, e_std = _mean_std(self._errors_all or [])

        eff_n = _effective_n_from_weights(self._weights_all or [])

        return {
            "kind": "adaboost",
            "task": "classification",
            "metric_name": self.metric_name,
            "base_algo": self.base_algo,
            "n_estimators": _safe_int(self.n_estimators),
            "learning_rate": _safe_float(self.learning_rate),
            "algorithm": self.algorithm,
            "vote": {
                "mean_margin": _safe_float(mean_margin),
                "mean_strength": _safe_float(mean_strength),
                "tie_rate": _safe_float(tie_rate),
                "margin_hist": {
                    "edges": [float(x) for x in (self._margin_hist_edges or []).tolist()],
                    "counts": [float(x) for x in (self._margin_hist_counts or []).tolist()],
                },
                "strength_hist": {
                    "edges": [float(x) for x in (self._strength_hist_edges or []).tolist()],
                    "counts": [float(x) for x in (self._strength_hist_counts or []).tolist()],
                },
            },
            "estimators": {
                "weight": {
                    "mean": _safe_float(w_mean),
                    "std": _safe_float(w_std),
                    "n": _safe_int(len(self._weights_all or [])),
                    "effective_n": _safe_float(eff_n) if eff_n is not None else None,
                },
                "error": {
                    "mean": _safe_float(e_mean),
                    "std": _safe_float(e_std),
                    "n": _safe_int(len(self._errors_all or [])),
                },
            },
        }
