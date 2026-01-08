# utils/postprocessing/ensembles/bagging_ensemble_reporting.py
"""Bagging ensemble reporting utilities (business-logic layer).

Goal: provide ensemble-specific insights across folds, similar to voting.

This module is backend/frontend agnostic: backend services can populate it,
but it can also be used from standalone scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .common import _safe_float, _safe_int, _mean_std

import numpy as np

def _vote_margin_and_strength(preds_row: Sequence[Any]) -> Tuple[float, float, bool]:
    """Compute vote margin and strength for a single sample (unweighted).

    Returns (margin, strength, is_tie_for_top).
      - margin: top_vote - second_vote (>= 0)
      - strength: top_vote / total_vote (in [0,1] if total_vote>0)
      - tie: True if top_vote == second_vote
    """
    counts: Dict[Any, float] = {}
    total = 0.0
    for p in preds_row:
        total += 1.0
        counts[p] = counts.get(p, 0.0) + 1.0

    if not counts or total <= 0:
        return 0.0, 0.0, True

    votes = sorted(counts.values(), reverse=True)
    top = votes[0]
    second = votes[1] if len(votes) > 1 else 0.0
    margin = max(0.0, float(top - second))
    strength = float(top / total) if total > 0 else 0.0
    tie = bool(np.isclose(top, second))
    return margin, strength, tie


def _oob_coverage_from_decision_function(oob_decision: Any) -> Optional[float]:
    """Estimate OOB coverage from sklearn-style oob_decision_function_.

    Typically shape: (n_train, n_classes) with NaNs for samples that never had OOB preds.
    Coverage = fraction of rows that have no NaNs.
    """
    try:
        a = np.asarray(oob_decision)
        if a.ndim < 2:
            return None
        ok = ~np.any(np.isnan(a), axis=1)
        if ok.size == 0:
            return None
        return float(np.mean(ok))
    except Exception:
        return None


@dataclass
class BaggingEnsembleReportAccumulator:
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
    _n_folds: int = 0

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

        # margin in unweighted voting: 0..m
        edges_margin = np.arange(-0.5, float(m) + 1.5, 1.0) if m > 0 else np.array([-0.5, 0.5])
        edges_strength = np.linspace(0.0, 1.0, num=21)

        # base estimator score distribution (usually 0..1; still safe if metric differs)
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
        """Add fold data.

        Parameters
        ----------
        base_preds:
            Array of shape (n_eval, n_estimators) with each estimator's predictions on the fold eval set.
            This is enough to compute agreement + vote margin/strength/ties.
        oob_score:
            model.oob_score_ for this fold, if available.
        oob_decision_function:
            model.oob_decision_function_ for this fold, if available (used for coverage).
        base_estimator_scores:
            Optional per-estimator scores computed on the eval set (same metric as overall),
            e.g. [score(est_i)] for i in estimators_. This is a distribution (not “est_1 across folds”).
        """
        self._n_folds += 1

        if oob_score is not None:
            self._oob_scores.append(float(oob_score))

        cov = _oob_coverage_from_decision_function(oob_decision_function)
        if cov is not None:
            self._oob_coverages.append(float(cov))

        P = np.asarray(base_preds)
        if P.ndim != 2:
            return

        n, m = int(P.shape[0]), int(P.shape[1])
        if n <= 0 or m <= 0:
            return

        # eval sample count
        self._n_total_eval += n

        # all-agree rate
        all_agree = np.all(P == P[:, [0]], axis=1)
        self._all_agree_count += int(np.sum(all_agree))

        # pairwise agreement counts
        if self._pairwise_same.shape == (m, m):
            for i in range(m):
                for j in range(i, m):
                    same = float(np.sum(P[:, i] == P[:, j]))
                    self._pairwise_same[i, j] += same
                    self._pairwise_same[j, i] += same if i != j else 0.0

        # vote margin/strength + tie rate
        margins = np.zeros(n, dtype=float)
        strengths = np.zeros(n, dtype=float)
        ties = 0
        for r in range(n):
            margin, strength, tie = _vote_margin_and_strength(P[r, :])
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

        # Optional base estimator score distribution
        if base_estimator_scores is not None:
            vals = [float(s) for s in base_estimator_scores if s is not None]
            if vals:
                self._base_scores_all.extend(vals)
                hh, _ = np.histogram(vals, bins=self._base_score_hist_edges)
                self._base_score_hist_counts += hh

    def finalize(self) -> Dict[str, Any]:
        denom = float(self._n_total_eval) if self._n_total_eval > 0 else 1.0

        # agreement
        pairwise = None
        pairwise_mean = None
        if self._pairwise_same.size and self.n_estimators > 0:
            pairwise = self._pairwise_same / denom
            m = pairwise.shape[0]
            if m > 1:
                mask = ~np.eye(m, dtype=bool)
                pairwise_mean = float(np.mean(pairwise[mask]))
            else:
                pairwise_mean = 1.0

        all_agree_rate = float(self._all_agree_count / denom) if denom else 0.0

        # vote stats
        n_total = float(self._n_total_eval) if self._n_total_eval > 0 else 1.0
        mean_margin = float(self._margin_sum / n_total)
        mean_strength = float(self._strength_sum / n_total)
        tie_rate = float(self._tie_count / n_total)

        # OOB stats
        oob_mean, oob_std = _mean_std(self._oob_scores or [])
        cov_mean, cov_std = _mean_std(self._oob_coverages or [])

        # base estimator score distribution (optional)
        base_mean, base_std = _mean_std(self._base_scores_all or [])

        # Keep payload reasonable: only ship full matrix when small
        MAX_MATRIX_ESTIMATORS = 100
        matrix_out = None
        labels_out = None
        if pairwise is not None and self.n_estimators <= MAX_MATRIX_ESTIMATORS:
            matrix_out = pairwise.tolist()
            labels_out = [f"est_{i+1}" for i in range(self.n_estimators)]

        report: Dict[str, Any] = {
            "kind": "bagging",
            "metric_name": self.metric_name,
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
                "score": _safe_float(oob_mean) if self._oob_scores else None,
                "score_std": _safe_float(oob_std) if self._oob_scores else None,
                "coverage_rate": _safe_float(cov_mean) if self._oob_coverages else None,
                "coverage_std": _safe_float(cov_std) if self._oob_coverages else None,
                "n_folds_with_oob": _safe_int(len(self._oob_scores)),
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
                    "edges": [float(x) for x in self._margin_hist_edges.tolist()],
                    "counts": [float(x) for x in self._margin_hist_counts.tolist()],
                },
                "strength_hist": {
                    "edges": [float(x) for x in self._strength_hist_edges.tolist()],
                    "counts": [float(x) for x in self._strength_hist_counts.tolist()],
                },
            },
            "base_estimator_scores": {
                "mean": _safe_float(base_mean) if self._base_scores_all else None,
                "std": _safe_float(base_std) if self._base_scores_all else None,
                "hist": {
                    "edges": [float(x) for x in self._base_score_hist_edges.tolist()],
                    "counts": [float(x) for x in self._base_score_hist_counts.tolist()],
                }
                if self._base_scores_all
                else None,
            },
            "meta": {
                "n_folds": _safe_int(self._n_folds),
                "n_eval_samples_total": _safe_int(self._n_total_eval),
            },
        }

        return report


__all__ = ["BaggingEnsembleReportAccumulator"]
