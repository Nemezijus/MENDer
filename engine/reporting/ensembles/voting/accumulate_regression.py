from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..common import _mean_std, _safe_float, _safe_int


@dataclass
class VotingEnsembleRegressorReportAccumulator:
    """Accumulate regression-specific insights for VotingRegressor across folds."""

    estimator_names: List[str]
    estimator_algos: List[str]
    metric_name: str
    weights: Optional[List[float]]

    _scores: Dict[str, List[float]]
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
        return cls(
            estimator_names=names,
            estimator_algos=algos,
            metric_name=str(metric_name),
            weights=w,
            _scores={n: [] for n in names},
            _y_true_parts=[],
            _y_ens_pred_parts=[],
            _y_base_pred_parts={n: [] for n in names},
        )

    def add_fold(
        self,
        *,
        y_true: np.ndarray,
        y_ensemble_pred: np.ndarray,
        base_preds: Dict[str, np.ndarray],
        base_scores: Dict[str, float],
    ) -> None:
        for name, s in base_scores.items():
            if name in self._scores:
                self._scores[name].append(float(s))

        self._y_true_parts.append(np.asarray(y_true, dtype=float).ravel())
        self._y_ens_pred_parts.append(np.asarray(y_ensemble_pred, dtype=float).ravel())
        for name, arr in base_preds.items():
            if name in self._y_base_pred_parts:
                self._y_base_pred_parts[name].append(np.asarray(arr, dtype=float).ravel())

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            y_true = np.asarray(y_true, dtype=float).ravel()
            y_pred = np.asarray(y_pred, dtype=float).ravel()
            if y_true.size == 0:
                return 0.0
            return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        except Exception:
            return 0.0

    @staticmethod
    def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            y_true = np.asarray(y_true, dtype=float).ravel()
            y_pred = np.asarray(y_pred, dtype=float).ravel()
            if y_true.size == 0:
                return 0.0
            return float(np.mean(np.abs(y_pred - y_true)))
        except Exception:
            return 0.0

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
        per_est = []
        best_name = None
        best_mean = None

        for name, algo in zip(self.estimator_names, self.estimator_algos):
            scores = self._scores.get(name, [])
            mean, std = _mean_std(scores)
            entry = {
                "name": name,
                "algo": algo,
                "fold_scores": [float(s) for s in scores] if scores else None,
                "mean": _safe_float(mean),
                "std": _safe_float(std),
                "n": len(scores),
            }
            per_est.append(entry)
            if best_mean is None or mean > best_mean:
                best_mean = mean
                best_name = name

        y_true = np.concatenate(self._y_true_parts) if self._y_true_parts else np.asarray([], dtype=float)
        y_ens = np.concatenate(self._y_ens_pred_parts) if self._y_ens_pred_parts else np.asarray([], dtype=float)

        # best base estimator by mean metric
        y_best = None
        if best_name is not None:
            parts = self._y_base_pred_parts.get(best_name, [])
            if parts:
                y_best = np.concatenate(parts)

        report: Dict[str, Any] = {
            "kind": "voting",
            "task": "regression",
            "metric_name": self.metric_name,
            "n_estimators": len(self.estimator_names),
            "weights": self.weights,
            "estimators": per_est,
            "best_estimator": {
                "name": best_name,
                "mean": _safe_float(best_mean) if best_mean is not None else None,
            },
            "errors": {
                "ensemble": {
                    "rmse": _safe_float(self._rmse(y_true, y_ens)) if y_true.size else None,
                    "mae": _safe_float(self._mae(y_true, y_ens)) if y_true.size else None,
                    "median_ae": _safe_float(self._median_ae(y_true, y_ens)) if y_true.size else None,
                },
                "best_base": {
                    "rmse": _safe_float(self._rmse(y_true, y_best)) if (y_true.size and y_best is not None) else None,
                    "mae": _safe_float(self._mae(y_true, y_best)) if (y_true.size and y_best is not None) else None,
                    "median_ae": _safe_float(self._median_ae(y_true, y_best)) if (y_true.size and y_best is not None) else None,
                },
            },
            "n_total": _safe_int(int(y_true.size)),
        }

        return report
