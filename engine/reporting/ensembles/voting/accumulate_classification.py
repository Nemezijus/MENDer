from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from engine.reporting.ensembles.accumulators.common_sections import (
    classification_effect_vs_best,
    concat_parts,
    finalize_pairwise_agreement,
    hist_add_inplace,
    margins_strengths_and_ties,
    update_all_agree_and_pairwise,
)

from engine.reporting.ensembles.accumulators import PerEstimatorFoldAccumulatorBase

from ..common import _safe_float, _safe_int


@dataclass
class VotingEnsembleReportAccumulator(PerEstimatorFoldAccumulatorBase):
    """Accumulate ensemble-specific insights across folds (VotingClassifier)."""

    estimator_names: List[str]
    estimator_algos: List[str]
    metric_name: str
    weights: Optional[List[float]]
    voting: str

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

        acc = cls(
            estimator_names=names,
            estimator_algos=algos,
            metric_name=str(metric_name),
            weights=w,
            voting=str(voting),
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

        self._y_true_parts.append(np.asarray(y_true))
        self._y_ens_pred_parts.append(np.asarray(y_ensemble_pred))
        for name, arr in base_preds.items():
            if name in self._y_base_pred_parts:
                self._y_base_pred_parts[name].append(np.asarray(arr))

        names = self.estimator_names
        pred_cols = [np.asarray(base_preds[n]) for n in names]
        P = np.stack(pred_cols, axis=1)  # (n, m)

        all_agree_count, n = update_all_agree_and_pairwise(base_preds=P, pairwise_same=self._pairwise_same)
        self._n_total += int(n)
        self._all_agree_count += int(all_agree_count)

        margins, strengths, ties = margins_strengths_and_ties(base_preds=P, weights=self.weights)
        self._tie_count += int(ties)
        self._margin_sum += float(np.sum(margins))
        self._strength_sum += float(np.sum(strengths))

        hist_add_inplace(counts=self._margin_hist_counts, edges=self._margin_hist_edges, values=margins)
        hist_add_inplace(counts=self._strength_hist_counts, edges=self._strength_hist_edges, values=strengths)

    def finalize(self) -> Dict[str, Any]:
        per_est, best = self._finalize_estimator_summaries()
        best_name = best.get("name") if best else None

        denom = float(self._n_total) if self._n_total > 0 else 1.0
        all_agree_rate = float(self._all_agree_count / denom) if denom else 0.0
        pairwise_mean, pairwise_mat = finalize_pairwise_agreement(pairwise_same=self._pairwise_same, n_total=self._n_total)

        n_total = float(self._n_total) if self._n_total > 0 else 1.0
        mean_margin = float(self._margin_sum / n_total)
        mean_strength = float(self._strength_sum / n_total)
        tie_rate = float(self._tie_count / n_total)

        corrected = harmed = disagreed = total = 0
        if best_name is not None and self._y_true_parts:
            y_true = concat_parts(self._y_true_parts)
            y_ens = concat_parts(self._y_ens_pred_parts)
            y_best = concat_parts(self._y_base_pred_parts.get(best_name, []))
            corrected, harmed, disagreed, total = classification_effect_vs_best(
                y_true=y_true, y_ensemble_pred=y_ens, y_best_pred=y_best
            )

        report: Dict[str, Any] = {
            "kind": "voting",
            "task": "classification",
            "metric_name": self.metric_name,
            "voting": self.voting,
            "n_estimators": len(self.estimator_names),
            "weights": self.weights,
            "estimators": per_est,
            "best_estimator": best,
            "agreement": {
                "all_agree_rate": _safe_float(all_agree_rate),
                "pairwise_mean_agreement": _safe_float(pairwise_mean) if pairwise_mean is not None else None,
                "labels": list(self.estimator_names),
                "matrix": pairwise_mat,
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
            "effect": {
                "corrected_vs_best": _safe_int(corrected),
                "harmed_vs_best": _safe_int(harmed),
                "disagreed_with_best": _safe_int(disagreed),
                "total": _safe_int(total),
            },
            "change_vs_best": {
                "best_name": (best.get("name") if best else None),
                "corrected": _safe_int(corrected),
                "harmed": _safe_int(harmed),
                "net": _safe_int(corrected - harmed),
            },

        }

        return report
