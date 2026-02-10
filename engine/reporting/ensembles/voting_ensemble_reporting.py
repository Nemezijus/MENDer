"""Ensemble reporting utilities.

These helpers live in the business-logic layer so they can be used by:
  - backend services
  - standalone scripts (no backend/frontend)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .common import _safe_float, _safe_int, _mean_std

import numpy as np

def _vote_margin_and_strength(
    preds_row: Sequence[Any],
    *,
    weights: Optional[Sequence[float]] = None,
) -> Tuple[float, float, bool]:
    """Compute vote margin and strength for a single sample.

    Returns (margin, strength, is_tie_for_top).
      - margin: top_vote - second_vote (>= 0)
      - strength: top_vote / total_vote (in [0,1] if total_vote>0)
      - tie: True if top_vote == second_vote

    Works for weighted or unweighted voting.
    """
    if weights is None:
        weights = [1.0] * len(preds_row)

    counts: Dict[Any, float] = {}
    total = 0.0
    for p, w in zip(preds_row, weights):
        w = float(w)
        total += w
        counts[p] = counts.get(p, 0.0) + w

    if not counts or total <= 0:
        return 0.0, 0.0, True

    votes = sorted(counts.values(), reverse=True)
    top = votes[0]
    second = votes[1] if len(votes) > 1 else 0.0
    margin = max(0.0, float(top - second))
    strength = float(top / total) if total > 0 else 0.0
    tie = bool(np.isclose(top, second))
    return margin, strength, tie


@dataclass
class VotingEnsembleReportAccumulator:
    """Accumulate ensemble-specific insights across folds."""

    estimator_names: List[str]
    estimator_algos: List[str]
    metric_name: str
    weights: Optional[List[float]]
    voting: str

    _scores: Dict[str, List[float]]

    _y_true_parts: List[np.ndarray]
    _y_ens_pred_parts: List[np.ndarray]
    _y_base_pred_parts: Dict[str, List[np.ndarray]]

    _all_agree_count: int
    _n_total: int
    _pairwise_same: np.ndarray

    _margin_sum: float
    _strength_sum: float
    _tie_count: int
    _margin_hist_counts: np.ndarray
    _margin_hist_edges: np.ndarray
    _strength_hist_counts: np.ndarray
    _strength_hist_edges: np.ndarray

    @classmethod
    def create(
        cls,
        *,
        estimator_names: Sequence[str],
        estimator_algos: Sequence[str],
        metric_name: str,
        weights: Optional[Sequence[float]],
        voting: str,
    ) -> "VotingEnsembleReportAccumulator":
        names = list(estimator_names)
        algos = list(estimator_algos)
        w = list(weights) if weights is not None else None

        m = len(names)
        max_margin = float(sum(w)) if w is not None else float(m)

        if w is None:
            edges_margin = np.arange(-0.5, max_margin + 1.5, 1.0)
        else:
            edges_margin = np.linspace(0.0, max_margin, num=21)

        edges_strength = np.linspace(0.0, 1.0, num=21)

        return cls(
            estimator_names=names,
            estimator_algos=algos,
            metric_name=str(metric_name),
            weights=w,
            voting=str(voting),
            _scores={n: [] for n in names},
            _y_true_parts=[],
            _y_ens_pred_parts=[],
            _y_base_pred_parts={n: [] for n in names},
            _all_agree_count=0,
            _n_total=0,
            _pairwise_same=np.zeros((m, m), dtype=float),
            _margin_sum=0.0,
            _strength_sum=0.0,
            _tie_count=0,
            _margin_hist_counts=np.zeros(len(edges_margin) - 1, dtype=float),
            _margin_hist_edges=edges_margin,
            _strength_hist_counts=np.zeros(len(edges_strength) - 1, dtype=float),
            _strength_hist_edges=edges_strength,
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

        self._y_true_parts.append(np.asarray(y_true))
        self._y_ens_pred_parts.append(np.asarray(y_ensemble_pred))
        for name, arr in base_preds.items():
            if name in self._y_base_pred_parts:
                self._y_base_pred_parts[name].append(np.asarray(arr))

        names = self.estimator_names
        pred_cols = [np.asarray(base_preds[n]) for n in names]
        P = np.stack(pred_cols, axis=1)  # (n, m)
        n = int(P.shape[0])
        m = int(P.shape[1])

        self._n_total += n

        all_agree = np.all(P == P[:, [0]], axis=1)
        self._all_agree_count += int(np.sum(all_agree))

        for i in range(m):
            for j in range(i, m):
                same = float(np.sum(P[:, i] == P[:, j]))
                self._pairwise_same[i, j] += same
                self._pairwise_same[j, i] += same if i != j else 0.0

        w = self.weights
        margins = np.zeros(n, dtype=float)
        strengths = np.zeros(n, dtype=float)
        ties = 0
        for r in range(n):
            margin, strength, tie = _vote_margin_and_strength(P[r, :], weights=w)
            margins[r] = margin
            strengths[r] = strength
            ties += 1 if tie else 0

        self._tie_count += ties
        self._margin_sum += float(np.sum(margins))
        self._strength_sum += float(np.sum(strengths))

        mh, _ = np.histogram(margins, bins=self._margin_hist_edges)
        sh, _ = np.histogram(strengths, bins=self._strength_hist_edges)
        self._margin_hist_counts += mh
        self._strength_hist_counts += sh

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

        denom = float(self._n_total) if self._n_total > 0 else 1.0
        pairwise = self._pairwise_same / denom
        pairwise_mean = None
        if pairwise.size:
            m = pairwise.shape[0]
            if m > 1:
                mask = ~np.eye(m, dtype=bool)
                pairwise_mean = float(np.mean(pairwise[mask]))
            else:
                pairwise_mean = 1.0

        all_agree_rate = float(self._all_agree_count / denom) if denom else 0.0

        n_total = float(self._n_total) if self._n_total > 0 else 1.0
        mean_margin = float(self._margin_sum / n_total)
        mean_strength = float(self._strength_sum / n_total)
        tie_rate = float(self._tie_count / n_total)

        corrected = harmed = disagreed = total = 0
        if best_name is not None and self._y_true_parts:
            y_true = np.concatenate(self._y_true_parts)
            y_ens = np.concatenate(self._y_ens_pred_parts)
            y_best = np.concatenate(self._y_base_pred_parts.get(best_name, []))
            if y_true.size and y_ens.size and y_best.size:
                total = int(y_true.size)
                best_ok = y_best == y_true
                ens_ok = y_ens == y_true
                corrected = int(np.sum((~best_ok) & ens_ok))
                harmed = int(np.sum(best_ok & (~ens_ok)))
                disagreed = int(np.sum(y_best != y_ens))

        report: Dict[str, Any] = {
            "kind": "voting",
            "task": "classification",
            "metric_name": self.metric_name,
            "voting": self.voting,
            "n_estimators": len(self.estimator_names),
            "weights": self.weights,
            "estimators": per_est,
            "best_estimator": {
                "name": best_name,
                "mean": _safe_float(best_mean) if best_mean is not None else None,
            },
            "agreement": {
                "all_agree_rate": _safe_float(all_agree_rate),
                "pairwise_mean_agreement": _safe_float(pairwise_mean) if pairwise_mean is not None else None,
                "labels": list(self.estimator_names),
                "matrix": pairwise.tolist() if pairwise is not None else None,
            },
            "vote": {
                "mean_margin": _safe_float(mean_margin),
                "mean_strength": _safe_float(mean_strength),
                "tie_rate": _safe_float(tie_rate),
                "margin_hist": {
                    "edges": [float(x) for x in self._margin_hist_edges.tolist()],
                    "counts": [float(x) for x in self._margin_hist_counts.tolist()],
                },
                "strength_hist": {
                    "edges": [float(x) for x in self._strength_hist_edges.tolist()],
                    "counts": [float(x) for x in self._strength_hist_counts.tolist()],
                },
            },
            "change_vs_best": {
                "best_name": best_name,
                "total": _safe_int(total),
                "corrected": _safe_int(corrected),
                "harmed": _safe_int(harmed),
                "net": _safe_int(corrected - harmed),
                "disagreed": _safe_int(disagreed),
            },
        }

        return report


__all__ = ["VotingEnsembleReportAccumulator", "VotingEnsembleRegressorReportAccumulator"]



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
