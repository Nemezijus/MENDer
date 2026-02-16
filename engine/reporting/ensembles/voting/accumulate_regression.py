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
        # existing summaries
        per_est = self._finalize_estimator_summaries()

        y_true = concat_parts(self._y_true_parts)
        y_ens = concat_parts(self._y_ens_pred_parts)

        n_total = int(y_true.shape[0]) if y_true is not None else 0
        denom = float(n_total) if n_total > 0 else 1.0

        # ensemble errors
        ens_rmse = rmse(y_true, y_ens) if n_total > 0 else None
        ens_mae = mae(y_true, y_ens) if n_total > 0 else None
        ens_med = float(np.median(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_ens, dtype=float)))) if n_total > 0 else None

        # best base errors (by RMSE)
        best_name = None
        best_rmse = None
        best_mae = None
        best_med = None

        # build pooled base prediction matrix for similarity + best-base
        names = list(self._y_base_parts.keys())
        names_used: List[str] = []
        P_cols = []
        for name in names:
            yb = concat_parts(self._y_base_parts.get(name, []))
            if yb is None or yb.shape[0] != n_total:
                continue
            P_cols.append(np.asarray(yb, dtype=float))
            names_used.append(name)
            e_rmse = rmse(y_true, yb) if n_total > 0 else None
            if e_rmse is not None and (best_rmse is None or e_rmse < best_rmse):
                best_rmse = float(e_rmse)
                best_mae = float(mae(y_true, yb))
                best_med = float(np.median(np.abs(np.asarray(y_true, dtype=float) - np.asarray(yb, dtype=float))))
                best_name = name

        # similarity (optional)
        similarity = {
            "pairwise_mean_corr": None,
            "pairwise_mean_absdiff": None,
            "prediction_spread_mean": None,
            "labels": None,
            "pairwise_corr": None,
            "pairwise_absdiff": None,
        }
        if P_cols and n_total > 0:
            P = np.column_stack(P_cols)
            m_est = int(P.shape[1])
            spread = np.std(P, axis=1)
            similarity["prediction_spread_mean"] = _safe_float(float(np.mean(spread)))

            C = np.corrcoef(P, rowvar=False)
            C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(C, 1.0)
            D = np.zeros((m_est, m_est), dtype=float)
            for i in range(m_est):
                for j in range(i, m_est):
                    d = np.abs(P[:, i] - P[:, j])
                    v = float(np.mean(d)) if d.size else 0.0
                    D[i, j] = v
                    D[j, i] = v

            if m_est > 1:
                mask = ~np.eye(m_est, dtype=bool)
                similarity["pairwise_mean_corr"] = _safe_float(float(np.mean(C[mask])))
                similarity["pairwise_mean_absdiff"] = _safe_float(float(np.mean(D[mask])))
            else:
                similarity["pairwise_mean_corr"] = _safe_float(1.0)
                similarity["pairwise_mean_absdiff"] = _safe_float(0.0)

            MAX_MATRIX_ESTIMATORS = 80
            if m_est <= MAX_MATRIX_ESTIMATORS:
                similarity["labels"] = names_used
                similarity["pairwise_corr"] = C.tolist()
                similarity["pairwise_absdiff"] = D.tolist()

        # best base score (by metric)
        best = select_best_estimator(list(per_est.values()))
        best_summary = (
            {
                "name": best["name"],
                "score": _safe_float(best["score_mean"]),
                "score_std": _safe_float(best["score_std"]),
            }
            if best
            else None
        )

        # gain vs best (using RMSE/MAE/median AE)
        gain = {
            "rmse_reduction": _safe_float(float(best_rmse - ens_rmse)) if (best_rmse is not None and ens_rmse is not None) else None,
            "mae_reduction": _safe_float(float(best_mae - ens_mae)) if (best_mae is not None and ens_mae is not None) else None,
            "median_ae_reduction": _safe_float(float(best_med - ens_med)) if (best_med is not None and ens_med is not None) else None,
            "r2": None,
        }

        return {
            "kind": "voting",
            "task": "regression",
            "metric_name": self.metric_name,
            "estimators": list(per_est.values()),
            "ensemble": {
                "score": _safe_float(self._ensemble_sum / max(1, self._n_folds)) if self._n_folds > 0 else None,
                "score_std": None,
            },
            "best_base": best_summary,
            "similarity": similarity,
            "errors": {
                "ensemble": {"rmse": _safe_float(ens_rmse), "mae": _safe_float(ens_mae), "median_ae": _safe_float(ens_med), "r2": None},
                "best_base": {"rmse": _safe_float(best_rmse), "mae": _safe_float(best_mae), "median_ae": _safe_float(best_med), "r2": None, "name": best_name},
                "gain_vs_best": gain,
                "n_total": _safe_int(n_total),
            },
            "meta": {
                "n_folds": _safe_int(self._n_folds),
                "n_eval_samples_total": _safe_int(n_total),
            },
        }
