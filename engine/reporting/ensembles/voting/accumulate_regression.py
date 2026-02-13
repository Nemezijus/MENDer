from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from engine.reporting.ensembles.accumulators import PerEstimatorFoldAccumulatorBase
from engine.reporting.ensembles.accumulators.common_sections import concat_parts, mae, rmse

from ..common import _safe_float, _safe_int


@dataclass
class VotingEnsembleRegressorReportAccumulator(PerEstimatorFoldAccumulatorBase):
    """Accumulate regression-specific insights for VotingRegressor across folds."""

    estimator_names: List[str]
    estimator_algos: List[str]
    metric_name: str
    weights: Optional[List[float]]

    _y_true_parts: List[np.ndarray]
    _y_ens_pred_parts: List[np.ndarray]
    _y_base_pred_parts: Dict[str, List[np.ndarray]]

    @classmethod
    def create(
        cls,
        *,
        estimator_names: Sequence[str],
        estimator_algos: Sequence[str],
        metric_name: str,
        weights: Optional[Sequence[float]],
    ) -> "VotingEnsembleRegressorReportAccumulator":
        names = list(estimator_names)
        algos = list(estimator_algos)
        w = list(weights) if weights is not None else None
        acc = cls(
            estimator_names=names,
            estimator_algos=algos,
            metric_name=str(metric_name),
            weights=w,
            _y_true_parts=[],
            _y_ens_pred_parts=[],
            _y_base_pred_parts={n: [] for n in names},
        )
        acc._init_scores()
        return acc

    def add_fold(
        self,
        *,
        y_true: np.ndarray,
        y_ensemble_pred: np.ndarray,
        base_preds: Dict[str, np.ndarray],
        base_scores: Dict[str, float],
    ) -> None:
        self._record_fold_scores(base_scores)

        self._y_true_parts.append(np.asarray(y_true, dtype=float).ravel())
        self._y_ens_pred_parts.append(np.asarray(y_ensemble_pred, dtype=float).ravel())
        for name, arr in base_preds.items():
            if name in self._y_base_pred_parts:
                self._y_base_pred_parts[name].append(np.asarray(arr, dtype=float).ravel())

    @staticmethod
    def _median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            y_true = np.asarray(y_true, dtype=float).ravel()
            y_pred = np.asarray(y_pred, dtype=float).ravel()
            if y_true.size == 0:
                return 0.0
            return float(np.median(np.abs(y_pred - y_true)))
        except Exception:
            return 0.0

    def finalize(self) -> Dict[str, Any]:
        per_est, best = self._finalize_estimator_summaries()
        best_name = best.get("name") if best else None

        y_true = concat_parts(self._y_true_parts, dtype=float)
        y_ens = concat_parts(self._y_ens_pred_parts, dtype=float)

        # best base estimator by mean metric
        y_best = None
        if best_name is not None:
            parts = self._y_base_pred_parts.get(best_name, [])
            if parts:
                y_best = concat_parts(parts, dtype=float)

        report: Dict[str, Any] = {
            "kind": "voting",
            "task": "regression",
            "metric_name": self.metric_name,
            "n_estimators": len(self.estimator_names),
            "weights": self.weights,
            "estimators": per_est,
            "best_estimator": best,
            "errors": {
                "ensemble": {
                    "rmse": _safe_float(rmse(y_true, y_ens)) if y_true.size else None,
                    "mae": _safe_float(mae(y_true, y_ens)) if y_true.size else None,
                    "median_ae": _safe_float(self._median_ae(y_true, y_ens)) if y_true.size else None,
                },
                "best_base": {
                    "rmse": _safe_float(rmse(y_true, y_best)) if (y_true.size and y_best is not None) else None,
                    "mae": _safe_float(mae(y_true, y_best)) if (y_true.size and y_best is not None) else None,
                    "median_ae": _safe_float(self._median_ae(y_true, y_best)) if (y_true.size and y_best is not None) else None,
                },
            },
            "n_total": _safe_int(int(y_true.size)),
        }

        return report
