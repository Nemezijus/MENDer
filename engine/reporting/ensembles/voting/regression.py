from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..common import _mean_std, _safe_float, _safe_int


@dataclass
class VotingEnsembleRegressorReportAccumulator:
    """Accumulate regression-specific insights for VotingRegressor across folds.

    We avoid classification concepts (vote margin, ties) and instead summarize:
      - per-estimator fold scores (using the selected metric)
      - prediction similarity / diversity (pairwise Pearson correlation, abs-diff)
      - error summaries (RMSE/MAE/median AE) for ensemble and best base estimator

    """

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
        # --- per-estimator metric summary ----------------------------------
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

            # Higher-is-better assumption for the chosen metric (consistent with evaluator)
            if best_mean is None or mean > best_mean:
                best_mean = mean
                best_name = name

        # --- concatenate predictions --------------------------------------
        y_true = np.concatenate(self._y_true_parts) if self._y_true_parts else np.asarray([], dtype=float)
        y_ens = np.concatenate(self._y_ens_pred_parts) if self._y_ens_pred_parts else np.asarray([], dtype=float)

        # Base prediction matrix for similarity / diversity
        P = None
        if y_true.size and self.estimator_names:
            cols = []
            for n in self.estimator_names:
                parts = self._y_base_pred_parts.get(n, [])
                if parts:
                    cols.append(np.concatenate(parts))
                else:
                    cols.append(np.full_like(y_true, np.nan, dtype=float))
            try:
                P = np.stack(cols, axis=1)  # (N, M)
            except Exception:
                P = None

        # --- similarity metrics -------------------------------------------
        pairwise_corr = None
        pairwise_absdiff = None
        pairwise_mean_corr = None
        pairwise_mean_absdiff = None
        pred_spread_mean = None

        if P is not None and P.size:
            M = P.shape[1]
            corr = np.eye(M, dtype=float)
            absd = np.zeros((M, M), dtype=float)

            for i in range(M):
                xi = P[:, i]
                for j in range(i, M):
                    xj = P[:, j]

                    # abs diff
                    try:
                        d = np.nanmean(np.abs(xi - xj))
                    except Exception:
                        d = 0.0
                    absd[i, j] = float(d)
                    absd[j, i] = float(d)

                    if i == j:
                        continue

                    # Pearson corr (robust to constant preds)
                    try:
                        xi0 = xi - np.nanmean(xi)
                        xj0 = xj - np.nanmean(xj)
                        denom = float(np.sqrt(np.nanmean(xi0**2) * np.nanmean(xj0**2)))
                        c = float(np.nanmean(xi0 * xj0) / denom) if denom > 0 else 0.0
                    except Exception:
                        c = 0.0
                    if not np.isfinite(c):
                        c = 0.0
                    corr[i, j] = c
                    corr[j, i] = c

            mask = ~np.eye(M, dtype=bool)
            if M > 1:
                pairwise_mean_corr = float(np.mean(corr[mask]))
                pairwise_mean_absdiff = float(np.mean(absd[mask]))
            else:
                pairwise_mean_corr = 1.0
                pairwise_mean_absdiff = 0.0

            pairwise_corr = corr.tolist()
            pairwise_absdiff = absd.tolist()

            # per-sample prediction spread (std across estimators)
            try:
                pred_spread_mean = float(np.nanmean(np.nanstd(P, axis=1)))
            except Exception:
                pred_spread_mean = 0.0

        # --- error summaries (ensemble vs best base) -----------------------
        ens_rmse = self._rmse(y_true, y_ens)
        ens_mae = self._mae(y_true, y_ens)
        ens_medae = self._median_ae(y_true, y_ens)

        best_rmse = best_mae = best_medae = None
        rmse_gain = mae_gain = medae_gain = None

        if best_name is not None and y_true.size:
            y_best_parts = self._y_base_pred_parts.get(best_name, [])
            if y_best_parts:
                y_best = np.concatenate(y_best_parts)
                best_rmse = self._rmse(y_true, y_best)
                best_mae = self._mae(y_true, y_best)
                best_medae = self._median_ae(y_true, y_best)

                rmse_gain = float(best_rmse - ens_rmse)
                mae_gain = float(best_mae - ens_mae)
                medae_gain = float(best_medae - ens_medae)

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
            "similarity": {
                "labels": list(self.estimator_names),
                "pairwise_corr": pairwise_corr,
                "pairwise_absdiff": pairwise_absdiff,
                "pairwise_mean_corr": _safe_float(pairwise_mean_corr) if pairwise_mean_corr is not None else None,
                "pairwise_mean_absdiff": _safe_float(pairwise_mean_absdiff) if pairwise_mean_absdiff is not None else None,
                "prediction_spread_mean": _safe_float(pred_spread_mean) if pred_spread_mean is not None else None,
            },
            "errors": {
                "n_total": _safe_int(int(y_true.size)),
                "ensemble": {
                    "rmse": _safe_float(ens_rmse),
                    "mae": _safe_float(ens_mae),
                    "median_ae": _safe_float(ens_medae),
                },
                "best_base": {
                    "name": best_name,
                    "rmse": _safe_float(best_rmse) if best_rmse is not None else None,
                    "mae": _safe_float(best_mae) if best_mae is not None else None,
                    "median_ae": _safe_float(best_medae) if best_medae is not None else None,
                },
                "gain_vs_best": {
                    "rmse_reduction": _safe_float(rmse_gain) if rmse_gain is not None else None,
                    "mae_reduction": _safe_float(mae_gain) if mae_gain is not None else None,
                    "median_ae_reduction": _safe_float(medae_gain) if medae_gain is not None else None,
                },
            },
        }

        return report

__all__ = ["VotingEnsembleRegressorReportAccumulator"]
