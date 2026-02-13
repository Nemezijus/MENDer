from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..common import _mean_std, _safe_float, _safe_int


@dataclass
class BaggingEnsembleRegressorReportAccumulator:
    """Accumulate regression-specific insights across folds (BaggingRegressor)."""

    metric_name: str
    base_algo: str

    n_estimators: int
    max_samples: Any
    max_features: Any
    bootstrap: bool
    bootstrap_features: bool
    oob_score_enabled: bool

    _n_folds: int = 0
    _oob_scores: List[float] = None

    _y_true_parts: List[np.ndarray] = None
    _y_ens_pred_parts: List[np.ndarray] = None
    _y_base_pred_parts: List[np.ndarray] = None
    _base_scores_all: List[float] = None

    @classmethod
    def create(
        cls,
        *,
        metric_name: str,
        base_algo: str,
        n_estimators: int,
        max_samples: Any,
        max_features: Any,
        bootstrap: bool,
        bootstrap_features: bool,
        oob_score_enabled: bool,
    ) -> "BaggingEnsembleRegressorReportAccumulator":
        return cls(
            metric_name=str(metric_name),
            base_algo=str(base_algo),
            n_estimators=int(n_estimators),
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bool(bootstrap),
            bootstrap_features=bool(bootstrap_features),
            oob_score_enabled=bool(oob_score_enabled),
            _oob_scores=[],
            _y_true_parts=[],
            _y_ens_pred_parts=[],
            _y_base_pred_parts=[],
            _base_scores_all=[],
        )

    def add_fold(
        self,
        *,
        y_true: np.ndarray,
        y_ensemble_pred: np.ndarray,
        base_preds: np.ndarray,
        oob_score: Optional[float] = None,
        base_estimator_scores: Optional[Sequence[float]] = None,
    ) -> None:
        self._n_folds += 1

        if oob_score is not None:
            self._oob_scores.append(float(oob_score))

        self._y_true_parts.append(np.asarray(y_true, dtype=float).ravel())
        self._y_ens_pred_parts.append(np.asarray(y_ensemble_pred, dtype=float).ravel())
        self._y_base_pred_parts.append(np.asarray(base_preds, dtype=float))  # (n, m)

        if base_estimator_scores:
            for s in base_estimator_scores:
                try:
                    self._base_scores_all.append(float(s))
                except Exception:
                    pass

    def finalize(self) -> Dict[str, Any]:
        y_true = np.concatenate(self._y_true_parts) if self._y_true_parts else np.asarray([], dtype=float)
        y_ens = np.concatenate(self._y_ens_pred_parts) if self._y_ens_pred_parts else np.asarray([], dtype=float)

        # base_preds concatenated as list of (n_fold, m)
        base_mat = None
        if self._y_base_pred_parts:
            base_mat = np.concatenate(self._y_base_pred_parts, axis=0)

        # basic error summaries
        def _rmse(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=float).ravel()
            b = np.asarray(b, dtype=float).ravel()
            if a.size == 0:
                return 0.0
            return float(np.sqrt(np.mean((b - a) ** 2)))

        def _mae(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=float).ravel()
            b = np.asarray(b, dtype=float).ravel()
            if a.size == 0:
                return 0.0
            return float(np.mean(np.abs(b - a)))

        ens_rmse = _rmse(y_true, y_ens)
        ens_mae = _mae(y_true, y_ens)

        base_rmse_mean = None
        base_mae_mean = None
        if base_mat is not None and y_true.size:
            rmses = []
            maes = []
            for i in range(base_mat.shape[1]):
                yp = base_mat[:, i]
                rmses.append(_rmse(y_true, yp))
                maes.append(_mae(y_true, yp))
            base_rmse_mean = float(np.mean(np.asarray(rmses))) if rmses else None
            base_mae_mean = float(np.mean(np.asarray(maes))) if maes else None

        oob_mean, oob_std = _mean_std(self._oob_scores or [])

        report: Dict[str, Any] = {
            "kind": "bagging",
            "task": "regression",
            "metric_name": self.metric_name,
            "base_algo": self.base_algo,
            "n_estimators": int(self.n_estimators),
            "bootstrap": bool(self.bootstrap),
            "bootstrap_features": bool(self.bootstrap_features),
            "oob": {
                "enabled": bool(self.oob_score_enabled),
                "scores": [float(x) for x in (self._oob_scores or [])] if (self._oob_scores) else None,
                "mean": _safe_float(oob_mean),
                "std": _safe_float(oob_std),
            },
            "errors": {
                "ensemble": {"rmse": _safe_float(ens_rmse), "mae": _safe_float(ens_mae)},
                "base_mean": {
                    "rmse": _safe_float(base_rmse_mean) if base_rmse_mean is not None else None,
                    "mae": _safe_float(base_mae_mean) if base_mae_mean is not None else None,
                },
            },
        }
        return report
