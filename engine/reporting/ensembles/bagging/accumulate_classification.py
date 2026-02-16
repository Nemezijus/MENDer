from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

from engine.reporting.ensembles.accumulators.common_sections import (
    finalize_pairwise_agreement,
    hist_add_inplace,
    margins_strengths_and_ties,
    update_all_agree_and_pairwise,
)

from engine.reporting.ensembles.accumulators import FoldAccumulatorBase

from ..common import _mean_std, _safe_float, _safe_int
from .helpers import oob_coverage_from_decision_function as _oob_coverage_from_decision_function


@dataclass
class BaggingEnsembleReportAccumulator(FoldAccumulatorBase):
    """Accumulate bagging-specific insights across folds."""

    metric_name: str
    base_algo: str

    n_estimators: int
    max_samples: Any
    max_features: Any
    bootstrap: bool
    bootstrap_features: bool
    oob_score_enabled: bool
    balanced: bool

    # optional extras (BalancedBagging)
    sampling_strategy: Optional[str] = None
    replacement: Optional[bool] = None

    # accumulators

    _oob_scores: List[float] = None
    _oob_coverages: List[float] = None

    _n_total_eval: int = 0
    _all_agree_count: int = 0
    _pairwise_same: np.ndarray = None

    _margin_sum: float = 0.0
    _strength_sum: float = 0.0
    _tie_count: int = 0
    _margin_hist_counts: np.ndarray = None
    _margin_hist_edges: np.ndarray = None
    _strength_hist_counts: np.ndarray = None
    _strength_hist_edges: np.ndarray = None

    # optional: distribution of base-estimator scores (if you compute them in service)
    _base_scores_all: List[float] = None
    _base_score_hist_counts: np.ndarray = None
    _base_score_hist_edges: np.ndarray = None

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
        balanced: bool,
        sampling_strategy: Optional[str] = None,
        replacement: Optional[bool] = None,
    ) -> "BaggingEnsembleReportAccumulator":
        m = int(n_estimators)

        edges_margin = np.arange(-0.5, float(m) + 1.5, 1.0) if m > 0 else np.array([-0.5, 0.5])
        edges_strength = np.linspace(0.0, 1.0, num=21)
        edges_score = np.linspace(0.0, 1.0, num=21)

        return cls(
            metric_name=str(metric_name),
            base_algo=str(base_algo),
            n_estimators=m,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bool(bootstrap),
            bootstrap_features=bool(bootstrap_features),
            oob_score_enabled=bool(oob_score_enabled),
            balanced=bool(balanced),
            sampling_strategy=sampling_strategy,
            replacement=replacement,
            _oob_scores=[],
            _oob_coverages=[],
            _pairwise_same=np.zeros((m, m), dtype=float) if m > 0 else np.zeros((0, 0), dtype=float),
            _margin_hist_edges=edges_margin,
            _margin_hist_counts=np.zeros(len(edges_margin) - 1, dtype=float),
            _strength_hist_edges=edges_strength,
            _strength_hist_counts=np.zeros(len(edges_strength) - 1, dtype=float),
            _base_scores_all=[],
            _base_score_hist_edges=edges_score,
            _base_score_hist_counts=np.zeros(len(edges_score) - 1, dtype=float),
        )

    def add_fold(
        self,
        *,
        base_preds: np.ndarray,
        oob_score: Optional[float] = None,
        oob_decision_function: Any = None,
        base_estimator_scores: Optional[Sequence[float]] = None,
    ) -> None:
        self._bump_fold()

        if oob_score is not None:
            self._oob_scores.append(float(oob_score))

        cov = _oob_coverage_from_decision_function(oob_decision_function)
        if cov is not None:
            self._oob_coverages.append(float(cov))

        P = np.asarray(base_preds)
        if P.ndim != 2:
            return

        all_agree_count, n = update_all_agree_and_pairwise(base_preds=P, pairwise_same=self._pairwise_same)
        self._n_total_eval += int(n)
        self._all_agree_count += int(all_agree_count)

        margins, strengths, ties = margins_strengths_and_ties(base_preds=P, weights=None)
        self._tie_count += int(ties)
        self._margin_sum += float(np.sum(margins))
        self._strength_sum += float(np.sum(strengths))

        hist_add_inplace(counts=self._margin_hist_counts, edges=self._margin_hist_edges, values=margins)
        hist_add_inplace(counts=self._strength_hist_counts, edges=self._strength_hist_edges, values=strengths)

        if base_estimator_scores:
            for s in base_estimator_scores:
                try:
                    self._base_scores_all.append(float(s))
                except Exception:
                    continue
            hist_add_inplace(
                counts=self._base_score_hist_counts,
                edges=self._base_score_hist_edges,
                values=np.asarray(base_estimator_scores, dtype=float),
            )

    def finalize(self) -> dict:
        n_total = int(self._n_total_eval)
        denom = float(n_total) if n_total > 0 else 1.0

        # agreement / diversity
        all_agree_rate = float(self._all_agree_count / denom) if denom else 0.0
        pairwise_mean, pairwise_mat = finalize_pairwise_agreement(pairwise_same=self._pairwise_same, n_total=n_total)

        # vote stats
        mean_margin = float(self._margin_sum / denom) if denom else 0.0
        mean_strength = float(self._strength_sum / denom) if denom else 0.0
        tie_rate = float(self._tie_count / denom) if denom else 0.0

        # OOB stats
        oob_mean, oob_std = _mean_std(self._oob_scores or [])
        cov_mean, cov_std = _mean_std(self._oob_coverages or [])

        # base estimator score distribution (optional)
        base_mean, base_std = _mean_std(self._base_scores_all or [])

        # keep payload size bounded (legacy behavior)
        MAX_MATRIX_ESTIMATORS = 100
        matrix_out = None
        labels_out = None
        if pairwise_mat is not None and int(self.n_estimators) <= MAX_MATRIX_ESTIMATORS:
            matrix_out = pairwise_mat
            labels_out = [f"est_{i+1}" for i in range(int(self.n_estimators))]

        report = {
            "kind": "bagging",
            "task": "classification",
            "metric_name": self.metric_name,

            # -----------------------------
            # legacy nested keys (frontend)
            # -----------------------------
            "bagging": {
                "base_algo": self.base_algo,
                "n_estimators": _safe_int(self.n_estimators),
                "max_samples": self.max_samples,
                "max_features": self.max_features,
                "bootstrap": bool(self.bootstrap),
                "bootstrap_features": bool(self.bootstrap_features),
                "oob_score_enabled": bool(self.oob_score_enabled),
                "balanced": bool(self.balanced),
                "sampling_strategy": self.sampling_strategy,
                "replacement": self.replacement,
            },
            "oob": {
                "score": _safe_float(oob_mean) if (self._oob_scores) else None,
                "score_std": _safe_float(oob_std) if (self._oob_scores) else None,
                "coverage_rate": _safe_float(cov_mean) if (self._oob_coverages) else None,
                "coverage_std": _safe_float(cov_std) if (self._oob_coverages) else None,
                "n_folds_with_oob": _safe_int(len(self._oob_scores or [])),

                # keep the new keys too (harmless)
                "enabled": bool(self.oob_score_enabled),
                "scores": [float(x) for x in (self._oob_scores or [])] if (self._oob_scores) else None,
                "score_mean": _safe_float(oob_mean),
                "score_mean_std": _safe_float(oob_std),
                "coverage_mean": _safe_float(cov_mean),
                "coverage_mean_std": _safe_float(cov_std),
            },
            "diversity": {
                "all_agree_rate": _safe_float(all_agree_rate),
                "pairwise_mean_agreement": _safe_float(pairwise_mean) if pairwise_mean is not None else None,
                "labels": labels_out,
                "matrix": matrix_out,
            },
            "vote": {
                "mean_margin": _safe_float(mean_margin),
                "mean_strength": _safe_float(mean_strength),
                "tie_rate": _safe_float(tie_rate),
                "margin_hist": {
                    "edges": [float(x) for x in (self._margin_hist_edges.tolist() if self._margin_hist_edges is not None else [])],
                    "counts": [float(x) for x in (self._margin_hist_counts.tolist() if self._margin_hist_counts is not None else [])],
                },
                "strength_hist": {
                    "edges": [float(x) for x in (self._strength_hist_edges.tolist() if self._strength_hist_edges is not None else [])],
                    "counts": [float(x) for x in (self._strength_hist_counts.tolist() if self._strength_hist_counts is not None else [])],
                },
            },
            "base_estimator_scores": {
                "mean": _safe_float(base_mean) if (self._base_scores_all) else None,
                "std": _safe_float(base_std) if (self._base_scores_all) else None,
                "hist": {
                    "edges": [float(x) for x in (self._base_score_hist_edges.tolist() if self._base_score_hist_edges is not None else [])],
                    "counts": [float(x) for x in (self._base_score_hist_counts.tolist() if self._base_score_hist_counts is not None else [])],
                }
                if (self._base_scores_all)
                else None,
            },
            "meta": {
                "n_folds": _safe_int(self._n_folds),
                "n_eval_samples_total": _safe_int(self._n_total_eval),
            },

            # -----------------------
            # new/flat keys (engine)
            # -----------------------
            "base_algo": self.base_algo,
            "n_estimators": _safe_int(self.n_estimators),
            "params": {
                "n_estimators": _safe_int(self.n_estimators),
                "max_samples": self.max_samples,
                "max_features": self.max_features,
                "bootstrap": bool(self.bootstrap),
                "bootstrap_features": bool(self.bootstrap_features),
                "oob_score_enabled": bool(self.oob_score_enabled),
                "balanced": bool(self.balanced),
                "sampling_strategy": self.sampling_strategy,
                "replacement": self.replacement,
            },
            "agreement": {
                "all_agree_rate": _safe_float(all_agree_rate),
                "pairwise_mean_agreement": _safe_float(pairwise_mean) if pairwise_mean is not None else None,
                "matrix": matrix_out,
            },
        }
        return report
