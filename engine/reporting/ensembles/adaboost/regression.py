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


@dataclass
class AdaBoostEnsembleRegressorReportAccumulator:
    """Accumulate regression-specific AdaBoost insights across folds."""

    metric_name: str
    base_algo: str

    n_estimators: int
    learning_rate: float
    loss: Optional[str] = None  # linear / square / exponential (sklearn)

    _n_folds: int = 0
    _n_eval_total: int = 0

    # weights/errors across folds
    _weights_all: List[float] = None
    _errors_all: List[float] = None
    _eff_n_all: List[float] = None

    # diagnostics
    _n_estimators_fitted_all: List[int] = None
    _n_nonzero_weights_all: List[int] = None
    _n_nontrivial_weights_all: List[int] = None
    _weight_eps: float = 1e-6
    _weight_mass_topk_all: Dict[int, List[float]] = None

    _weight_hist_edges: np.ndarray = None
    _weight_hist_counts: np.ndarray = None
    _error_hist_edges: np.ndarray = None
    _error_hist_counts: np.ndarray = None

    # diversity/similarity matrices (weighted by n_eval)
    _corr_sum: np.ndarray = None
    _absdiff_sum: np.ndarray = None
    _pred_spread_sum: float = 0.0

    # errors (weighted by n_eval)
    _ens_rmse_sum: float = 0.0
    _ens_mae_sum: float = 0.0
    _ens_med_sum: float = 0.0

    _best_rmse_sum: float = 0.0
    _best_mae_sum: float = 0.0
    _best_med_sum: float = 0.0

    _gain_rmse_sum: float = 0.0
    _gain_mae_sum: float = 0.0
    _gain_med_sum: float = 0.0

    # optional: base estimator score distribution
    _base_scores_all: List[float] = None

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
        n = int(n_estimators)
        weight_edges = np.linspace(0.0, 5.0, num=26)
        error_edges = np.linspace(0.0, 1.0, num=21)

        return cls(
            metric_name=str(metric_name),
            base_algo=str(base_algo),
            n_estimators=n,
            learning_rate=float(learning_rate),
            loss=str(loss) if loss is not None else None,
            _weights_all=[],
            _errors_all=[],
            _eff_n_all=[],
            _n_estimators_fitted_all=[],
            _n_nonzero_weights_all=[],
            _n_nontrivial_weights_all=[],
            _weight_eps=1e-6,
            _weight_mass_topk_all={5: [], 10: [], 20: []},
            _weight_hist_edges=weight_edges,
            _weight_hist_counts=np.zeros(len(weight_edges) - 1, dtype=float),
            _error_hist_edges=error_edges,
            _error_hist_counts=np.zeros(len(error_edges) - 1, dtype=float),
            _corr_sum=np.zeros((n, n), dtype=float) if n > 0 else np.zeros((0, 0), dtype=float),
            _absdiff_sum=np.zeros((n, n), dtype=float) if n > 0 else np.zeros((0, 0), dtype=float),
            _base_scores_all=[],
        )

    def add_fold(
        self,
        *,
        y_true: np.ndarray,
        ensemble_pred: np.ndarray,
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
                self._weight_mass_topk_all.setdefault(kk, []).append(float(v))

        P = np.asarray(base_preds, dtype=float)
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

        # similarity/diversity
        spread = np.std(P, axis=1)
        self._pred_spread_sum += float(np.sum(spread))

        C = _corr_matrix_safe(P)
        D = _mean_absdiff_matrix(P)
        if self._corr_sum.shape == (m, m):
            self._corr_sum += C * float(n_eval)
        if self._absdiff_sum.shape == (m, m):
            self._absdiff_sum += D * float(n_eval)

        # weights/errors stats
        w_pos = w[w > 0]
        if w_pos.size:
            self._weights_all.extend([float(x) for x in w_pos.tolist()])
            w_clip = np.clip(w_pos, self._weight_hist_edges[0], self._weight_hist_edges[-1] - 1e-12)
            h, _ = np.histogram(w_clip, bins=self._weight_hist_edges)
            self._weight_hist_counts += h
            # effective n
            s1 = float(np.sum(w_pos))
            s2 = float(np.sum(w_pos * w_pos))
            eff = (s1 * s1) / s2 if s2 > 0 else 0.0
            self._eff_n_all.append(float(eff))

        if estimator_errors is not None:
            e = np.asarray(estimator_errors, dtype=float)
            e = e[np.isfinite(e)]
            if e.size:
                self._errors_all.extend([float(x) for x in e.tolist()])
                e_clip = np.clip(e, self._error_hist_edges[0], self._error_hist_edges[-1] - 1e-12)
                h, _ = np.histogram(e_clip, bins=self._error_hist_edges)
                self._error_hist_counts += h

        # ensemble vs best weak learner
        ens_err = _regression_error_stats(y_true, ensemble_pred)
        self._ens_rmse_sum += ens_err["rmse"] * float(n_eval)
        self._ens_mae_sum += ens_err["mae"] * float(n_eval)
        self._ens_med_sum += ens_err["median_ae"] * float(n_eval)

        yt = np.asarray(y_true, dtype=float).ravel()
        best_rmse = None
        best_mae = None
        best_med = None
        for j in range(m):
            e = _regression_error_stats(yt, P[:, j])
            if best_rmse is None or e["rmse"] < best_rmse:
                best_rmse, best_mae, best_med = e["rmse"], e["mae"], e["median_ae"]

        if best_rmse is None:
            best_rmse = best_mae = best_med = 0.0

        self._best_rmse_sum += float(best_rmse) * float(n_eval)
        self._best_mae_sum += float(best_mae) * float(n_eval)
        self._best_med_sum += float(best_med) * float(n_eval)

        self._gain_rmse_sum += float(best_rmse - ens_err["rmse"]) * float(n_eval)
        self._gain_mae_sum += float(best_mae - ens_err["mae"]) * float(n_eval)
        self._gain_med_sum += float(best_med - ens_err["median_ae"]) * float(n_eval)

        if base_estimator_scores is not None:
            vals = [float(s) for s in base_estimator_scores if s is not None]
            if vals:
                self._base_scores_all.extend(vals)

    def finalize(self) -> Dict[str, Any]:
        denom = float(self._n_eval_total) if self._n_eval_total > 0 else 1.0

        # matrices
        MAX_MATRIX_ESTIMATORS = 80
        corr_out = None
        absdiff_out = None
        labels_out = None
        pairwise_mean_corr = None
        pairwise_mean_absdiff = None

        if self.n_estimators > 0 and self._corr_sum.size:
            C = self._corr_sum / denom
            D = self._absdiff_sum / denom
            m = C.shape[0]
            if m > 1:
                mask = ~np.eye(m, dtype=bool)
                pairwise_mean_corr = float(np.mean(C[mask]))
                pairwise_mean_absdiff = float(np.mean(D[mask]))
            else:
                pairwise_mean_corr = 1.0
                pairwise_mean_absdiff = 0.0

            if self.n_estimators <= MAX_MATRIX_ESTIMATORS:
                corr_out = C.tolist()
                absdiff_out = D.tolist()
                labels_out = [f"est_{i+1}" for i in range(self.n_estimators)]

        pred_spread_mean = float(self._pred_spread_sum / denom) if denom else 0.0

        # weights/errors summaries
        w_mean = float(np.mean(self._weights_all)) if self._weights_all else 0.0
        w_std = float(np.std(self._weights_all)) if self._weights_all else 0.0
        e_mean = float(np.mean(self._errors_all)) if self._errors_all else 0.0
        e_std = float(np.std(self._errors_all)) if self._errors_all else 0.0
        eff_mean = float(np.mean(self._eff_n_all)) if self._eff_n_all else 0.0
        eff_std = float(np.std(self._eff_n_all)) if self._eff_n_all else 0.0

        # stage diagnostics
        fitted_mean = float(np.mean(self._n_estimators_fitted_all)) if self._n_estimators_fitted_all else 0.0
        fitted_std = float(np.std(self._n_estimators_fitted_all)) if self._n_estimators_fitted_all else 0.0
        nz_mean = float(np.mean(self._n_nonzero_weights_all)) if self._n_nonzero_weights_all else 0.0
        nz_std = float(np.std(self._n_nonzero_weights_all)) if self._n_nonzero_weights_all else 0.0
        nt_mean = float(np.mean(self._n_nontrivial_weights_all)) if self._n_nontrivial_weights_all else 0.0
        nt_std = float(np.std(self._n_nontrivial_weights_all)) if self._n_nontrivial_weights_all else 0.0

        topk_mean = {}
        topk_std = {}
        if self._weight_mass_topk_all:
            for k, vals in self._weight_mass_topk_all.items():
                if not vals:
                    continue
                topk_mean[int(k)] = float(np.mean(vals))
                topk_std[int(k)] = float(np.std(vals))

        # errors (weighted)
        ens_rmse = float(self._ens_rmse_sum / denom)
        ens_mae = float(self._ens_mae_sum / denom)
        ens_med = float(self._ens_med_sum / denom)

        best_rmse = float(self._best_rmse_sum / denom)
        best_mae = float(self._best_mae_sum / denom)
        best_med = float(self._best_med_sum / denom)

        gain_rmse = float(self._gain_rmse_sum / denom)
        gain_mae = float(self._gain_mae_sum / denom)
        gain_med = float(self._gain_med_sum / denom)

        bs_mean = float(np.mean(self._base_scores_all)) if self._base_scores_all else 0.0
        bs_std = float(np.std(self._base_scores_all)) if self._base_scores_all else 0.0

        return {
            "kind": "adaboost",
            "task": "regression",
            "metric_name": self.metric_name,
            "adaboost": {
                "base_algo": self.base_algo,
                "n_estimators": _safe_int(self.n_estimators),
                "learning_rate": _safe_float(self.learning_rate),
                "loss": self.loss,
            },
            "diversity": {
                "pairwise_mean_corr": _safe_float(pairwise_mean_corr) if pairwise_mean_corr is not None else None,
                "pairwise_mean_absdiff": _safe_float(pairwise_mean_absdiff) if pairwise_mean_absdiff is not None else None,
                "prediction_spread_mean": _safe_float(pred_spread_mean),
                "labels": labels_out,
                "pairwise_corr": corr_out,
                "pairwise_absdiff": absdiff_out,
            },
            "weights": {
                "mean": _safe_float(w_mean) if self._weights_all else None,
                "std": _safe_float(w_std) if self._weights_all else None,
                "effective_n_mean": _safe_float(eff_mean) if self._eff_n_all else None,
                "effective_n_std": _safe_float(eff_std) if self._eff_n_all else None,
                "hist": {
                    "edges": [float(x) for x in self._weight_hist_edges.tolist()],
                    "counts": [float(x) for x in self._weight_hist_counts.tolist()],
                } if self._weights_all else None,
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
                } if self._errors_all else None,
            },
            "model_errors": {
                "ensemble": {"rmse": _safe_float(ens_rmse), "mae": _safe_float(ens_mae), "median_ae": _safe_float(ens_med)},
                "best_base": {"rmse": _safe_float(best_rmse), "mae": _safe_float(best_mae), "median_ae": _safe_float(best_med)},
                "gain_vs_best": {"rmse_reduction": _safe_float(gain_rmse), "mae_reduction": _safe_float(gain_mae), "median_ae_reduction": _safe_float(gain_med), "r2": None},
                "n_total": _safe_int(self._n_eval_total),
            },
            "base_estimator_scores": {
                "mean": _safe_float(bs_mean) if self._base_scores_all else None,
                "std": _safe_float(bs_std) if self._base_scores_all else None,
                "hist": _base_score_hist(self._base_scores_all) if self._base_scores_all else None,
            },
            "meta": {
                "n_folds": _safe_int(self._n_folds),
                "n_eval_samples_total": _safe_int(self._n_eval_total),
            },
        }

__all__ = ["AdaBoostEnsembleRegressorReportAccumulator"]
