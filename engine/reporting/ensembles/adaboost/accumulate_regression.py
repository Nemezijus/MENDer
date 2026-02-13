from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..common import _mean_std, _safe_float, _safe_int
from .helpers import (
    _effective_n_from_weights,
    _hist_add,
    _hist_init,
)

from engine.reporting.ensembles.accumulators import FoldAccumulatorBase


@dataclass
class AdaBoostEnsembleRegressorReportAccumulator(FoldAccumulatorBase):
    """Accumulate regression-specific AdaBoost insights across folds."""

    metric_name: str
    base_algo: str

    n_estimators: int
    learning_rate: float
    loss: Optional[str] = None  # linear / square / exponential (sklearn)

    _n_eval_total: int = 0

    # weights/errors across folds
    _weights_all: List[float] = None
    _errors_all: List[float] = None

    # base estimator score distribution (optional)
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
        loss: Optional[str] = None,
    ) -> "AdaBoostEnsembleRegressorReportAccumulator":
        edges_score = np.linspace(0.0, 1.0, num=21)
        return cls(
            metric_name=str(metric_name),
            base_algo=str(base_algo),
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            loss=(str(loss) if loss is not None else None),
            _weights_all=[],
            _errors_all=[],
            _base_scores_all=[],
            _base_score_hist_edges=edges_score,
            _base_score_hist_counts=np.zeros(len(edges_score) - 1, dtype=float),
        )

    def add_fold(
        self,
        *,
        estimator_weights: Optional[Sequence[float]] = None,
        estimator_errors: Optional[Sequence[float]] = None,
        base_estimator_scores: Optional[Sequence[float]] = None,
        n_eval: Optional[int] = None,
    ) -> None:
        self._bump_fold()

        if n_eval is not None:
            self._n_eval_total += int(n_eval)

        if estimator_weights:
            self._weights_all.extend([float(x) for x in estimator_weights])
        if estimator_errors:
            self._errors_all.extend([float(x) for x in estimator_errors])

        if base_estimator_scores:
            vals = [float(x) for x in base_estimator_scores]
            self._base_scores_all.extend(vals)
            _hist_add(self._base_score_hist_counts, self._base_score_hist_edges, vals)

    def finalize(self) -> Dict[str, Any]:
        w_mean, w_std = _mean_std(self._weights_all)
        e_mean, e_std = _mean_std(self._errors_all)

        eff_n = _effective_n_from_weights(self._weights_all) if self._weights_all else None

        score_mean, score_std = _mean_std(self._base_scores_all)

        report: Dict[str, Any] = {
            "kind": "adaboost",
            "task": "regression",
            "metric_name": self.metric_name,
            "base_algo": self.base_algo,
            "n_estimators": _safe_int(self.n_estimators),
            "learning_rate": _safe_float(self.learning_rate),
            "loss": self.loss,
            "n_folds": _safe_int(self._n_folds),
            "n_eval_total": _safe_int(self._n_eval_total),
            "estimator_weights": {
                "mean": _safe_float(w_mean),
                "std": _safe_float(w_std),
                "effective_n": _safe_float(eff_n) if eff_n is not None else None,
            },
            "estimator_errors": {
                "mean": _safe_float(e_mean),
                "std": _safe_float(e_std),
            },
            "base_estimator_scores": {
                "n": _safe_int(len(self._base_scores_all)),
                "mean": _safe_float(score_mean),
                "std": _safe_float(score_std),
                "hist": {
                    "edges": [float(x) for x in self._base_score_hist_edges.tolist()],
                    "counts": [float(x) for x in self._base_score_hist_counts.tolist()],
                },
            },
        }

        return report
