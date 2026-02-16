from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from engine.reporting.ensembles.accumulators import FoldAccumulatorBase
from engine.reporting.ensembles.common import _mean_std, _safe_float, _safe_int

from .helpers import (
    base_score_hist,
    corr_matrix_safe,
    mean_absdiff_matrix,
    oob_coverage_from_prediction,
    r2_score_safe,
    regression_error_stats,
)


@dataclass
class BaggingEnsembleRegressorReportAccumulator(FoldAccumulatorBase):
    """Accumulate regression-specific bagging insights across folds.

    This mirrors the legacy (pre-refactor) report schema expected by the frontend.
    """

    metric_name: str
    base_algo: str

    n_estimators: int
    max_samples: Any
    max_features: Any
    bootstrap: bool
    bootstrap_features: bool
    oob_score_enabled: bool

    _n_total_eval: int = 0

    _oob_scores: List[float] = None
    _oob_coverages: List[float] = None

    _pred_spread_sum: float = 0.0
    _corr_sum: np.ndarray = None
    _absdiff_sum: np.ndarray = None

    _ens_rmse_sum: float = 0.0
    _ens_mae_sum: float = 0.0
    _ens_med_sum: float = 0.0
    _ens_r2_sum: float = 0.0

    _best_rmse_sum: float = 0.0
    _best_mae_sum: float = 0.0
    _best_med_sum: float = 0.0
    _best_r2_sum: float = 0.0

    _gain_rmse_sum: float = 0.0
    _gain_mae_sum: float = 0.0
    _gain_med_sum: float = 0.0
    _gain_r2_sum: float = 0.0

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
        m = int(n_estimators)
        return cls(
            metric_name=str(metric_name),
            base_algo=str(base_algo),
            n_estimators=m,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bool(bootstrap),
            bootstrap_features=bool(bootstrap_features),
            oob_score_enabled=bool(oob_score_enabled),
            _oob_scores=[],
            _oob_coverages=[],
            _corr_sum=np.zeros((m, m), dtype=float) if m > 0 else np.zeros((0, 0), dtype=float),
            _absdiff_sum=np.zeros((m, m), dtype=float) if m > 0 else np.zeros((0, 0), dtype=float),
            _base_scores_all=[],
        )

    def add_fold(
        self,
        *,
        y_true: np.ndarray,
        ensemble_pred: Optional[np.ndarray] = None,
        y_ensemble_pred: Optional[np.ndarray] = None,
        base_preds: np.ndarray,
        oob_score: Optional[float] = None,
        oob_prediction: Any = None,
        base_estimator_scores: Optional[Sequence[float]] = None,
    ) -> None:
        """Add one fold's worth of diagnostics.

        Backend historically calls this with `ensemble_pred=...`, so we support both
        `ensemble_pred` and `y_ensemble_pred` names.
        """
        self._bump_fold()

        if oob_score is not None:
            self._oob_scores.append(float(oob_score))

        cov = oob_coverage_from_prediction(oob_prediction)
        if cov is not None:
            self._oob_coverages.append(float(cov))

        P = np.asarray(base_preds, dtype=float)
        if P.ndim != 2:
            return
        n, m = int(P.shape[0]), int(P.shape[1])
        if n <= 0 or m <= 0:
            return

        ens = ensemble_pred if ensemble_pred is not None else y_ensemble_pred
        if ens is None:
            return

        self._n_total_eval += n

        # similarity/diversity matrices
        spread = np.std(P, axis=1)
        self._pred_spread_sum += float(np.sum(spread))

        C = corr_matrix_safe(P)
        D = mean_absdiff_matrix(P)
        if self._corr_sum is not None and self._corr_sum.shape == (m, m):
            self._corr_sum += C * float(n)
        if self._absdiff_sum is not None and self._absdiff_sum.shape == (m, m):
            self._absdiff_sum += D * float(n)

        # errors
        ens_err = regression_error_stats(y_true, ens)
        ens_r2 = r2_score_safe(y_true, ens)
        self._ens_rmse_sum += ens_err["rmse"] * float(n)
        self._ens_mae_sum += ens_err["mae"] * float(n)
        self._ens_med_sum += ens_err["median_ae"] * float(n)
        self._ens_r2_sum += float(ens_r2) * float(n)

        yt = np.asarray(y_true, dtype=float).ravel()
        best_rmse = None
        best_mae = None
        best_med = None
        best_r2 = None
        for j in range(m):
            e = regression_error_stats(yt, P[:, j])
            r2j = r2_score_safe(yt, P[:, j])
            if best_rmse is None or e["rmse"] < best_rmse:
                best_rmse, best_mae, best_med = e["rmse"], e["mae"], e["median_ae"]
            if best_r2 is None or float(r2j) > float(best_r2):
                best_r2 = float(r2j)

        if best_rmse is None:
            best_rmse = best_mae = best_med = 0.0
        if best_r2 is None:
            best_r2 = 0.0

        self._best_rmse_sum += float(best_rmse) * float(n)
        self._best_mae_sum += float(best_mae) * float(n)
        self._best_med_sum += float(best_med) * float(n)
        self._best_r2_sum += float(best_r2) * float(n)

        self._gain_rmse_sum += float(best_rmse - ens_err["rmse"]) * float(n)
        self._gain_mae_sum += float(best_mae - ens_err["mae"]) * float(n)
        self._gain_med_sum += float(best_med - ens_err["median_ae"]) * float(n)
        self._gain_r2_sum += float(ens_r2 - best_r2) * float(n)

        if base_estimator_scores is not None:
            vals = [float(s) for s in base_estimator_scores if s is not None]
            if vals:
                self._base_scores_all.extend(vals)

    def finalize(self) -> Dict[str, Any]:
        denom = float(self._n_total_eval) if self._n_total_eval > 0 else 1.0

        oob_mean, oob_std = _mean_std(self._oob_scores or [])
        cov_mean, cov_std = _mean_std(self._oob_coverages or [])

        MAX_MATRIX_ESTIMATORS = 80
        corr_out = None
        absdiff_out = None
        labels_out = None
        pairwise_mean_corr = None
        pairwise_mean_absdiff = None

        if int(self.n_estimators) > 0 and self._corr_sum is not None and self._corr_sum.size:
            C = self._corr_sum / denom
            D = self._absdiff_sum / denom

            m = int(C.shape[0])
            if m > 1:
                mask = ~np.eye(m, dtype=bool)
                pairwise_mean_corr = float(np.mean(C[mask]))
                pairwise_mean_absdiff = float(np.mean(D[mask]))
            else:
                pairwise_mean_corr = 1.0
                pairwise_mean_absdiff = 0.0

            if int(self.n_estimators) <= MAX_MATRIX_ESTIMATORS:
                corr_out = C.tolist()
                absdiff_out = D.tolist()
                labels_out = [f"est_{i+1}" for i in range(int(self.n_estimators))]

        pred_spread_mean = float(self._pred_spread_sum / denom) if denom else 0.0

        ens_rmse = float(self._ens_rmse_sum / denom)
        ens_mae = float(self._ens_mae_sum / denom)
        ens_med = float(self._ens_med_sum / denom)
        ens_r2 = float(self._ens_r2_sum / denom)

        best_rmse = float(self._best_rmse_sum / denom)
        best_mae = float(self._best_mae_sum / denom)
        best_med = float(self._best_med_sum / denom)
        best_r2 = float(self._best_r2_sum / denom)

        gain_rmse = float(self._gain_rmse_sum / denom)
        gain_mae = float(self._gain_mae_sum / denom)
        gain_med = float(self._gain_med_sum / denom)
        gain_r2 = float(self._gain_r2_sum / denom)

        base_mean, base_std = _mean_std(self._base_scores_all or [])

        report: Dict[str, Any] = {
            "kind": "bagging",
            "task": "regression",
            "metric_name": self.metric_name,
            "bagging": {
                "base_algo": self.base_algo,
                "n_estimators": _safe_int(self.n_estimators),
                "max_samples": self.max_samples,
                "max_features": self.max_features,
                "bootstrap": bool(self.bootstrap),
                "bootstrap_features": bool(self.bootstrap_features),
                "oob_score_enabled": bool(self.oob_score_enabled),
                "balanced": False,
                "sampling_strategy": None,
                "replacement": None,
            },
            "oob": {
                "score": _safe_float(oob_mean) if self._oob_scores else None,
                "score_std": _safe_float(oob_std) if self._oob_scores else None,
                "coverage_rate": _safe_float(cov_mean) if self._oob_coverages else None,
                "coverage_std": _safe_float(cov_std) if self._oob_coverages else None,
                "n_folds_with_oob": _safe_int(len(self._oob_scores or [])),
            },
            "diversity": {
                "pairwise_mean_corr": _safe_float(pairwise_mean_corr) if pairwise_mean_corr is not None else None,
                "pairwise_mean_absdiff": _safe_float(pairwise_mean_absdiff) if pairwise_mean_absdiff is not None else None,
                "prediction_spread_mean": _safe_float(pred_spread_mean),
                "labels": labels_out,
                "pairwise_corr": corr_out,
                "pairwise_absdiff": absdiff_out,
            },
            "errors": {
                "ensemble": {"rmse": _safe_float(ens_rmse), "mae": _safe_float(ens_mae), "median_ae": _safe_float(ens_med), "r2": _safe_float(ens_r2)},
                "best_base": {"rmse": _safe_float(best_rmse), "mae": _safe_float(best_mae), "median_ae": _safe_float(best_med), "r2": _safe_float(best_r2)},
                "gain_vs_best": {"rmse": _safe_float(gain_rmse), "mae": _safe_float(gain_mae), "median_ae": _safe_float(gain_med), "r2": _safe_float(gain_r2)},
                "n_total": _safe_int(self._n_total_eval),
            },
            "base_estimator_scores": {
                "mean": _safe_float(base_mean) if self._base_scores_all else None,
                "std": _safe_float(base_std) if self._base_scores_all else None,
                "hist": base_score_hist(self._base_scores_all) if self._base_scores_all else None,
            },
            "meta": {
                "n_folds": _safe_int(self._n_folds),
                "n_eval_samples_total": _safe_int(self._n_total_eval),
            },
        }
        return report