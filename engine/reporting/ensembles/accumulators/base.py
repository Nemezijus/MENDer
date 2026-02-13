from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from engine.reporting.ensembles.common import _mean_std, _safe_float


@dataclass
class FoldAccumulatorBase:
    """Minimal shared base for fold-based accumulators.

    The goal is not to be clever — just to standardize the most common book-keeping
    fields so accumulator implementations stay consistent.
    """

    _n_folds: int = 0

    def _bump_fold(self) -> None:
        self._n_folds += 1


@dataclass
class PerEstimatorFoldAccumulatorBase(FoldAccumulatorBase):
    """Shared base for accumulators that track per-estimator fold scores.

    Many ensemble report accumulators repeat the same scaffolding:
      - create(): init per-estimator score lists
      - add_fold(): append base scores
      - finalize(): compute mean/std per estimator + pick the best

    Centralizing that book-keeping prevents drift when you add new common
    fields (e.g. additional summary stats) across families/tasks.
    """

    estimator_names: List[str] = None
    estimator_algos: List[str] = None
    metric_name: str = ""

    _scores: Dict[str, List[float]] = None

    def _init_scores(self) -> None:
        names = list(self.estimator_names or [])
        self._scores = {n: [] for n in names}

    def _record_fold_scores(self, base_scores: Dict[str, float]) -> None:
        if not base_scores or not self._scores:
            return
        for name, s in base_scores.items():
            if name in self._scores:
                try:
                    self._scores[name].append(float(s))
                except Exception:
                    continue

    def _finalize_estimator_summaries(self) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        per_est: List[Dict[str, Any]] = []
        best_name: Optional[str] = None
        best_mean: Optional[float] = None

        for name, algo in zip(self.estimator_names or [], self.estimator_algos or []):
            scores = (self._scores or {}).get(name, [])
            mean, std = _mean_std(scores)
            per_est.append(
                {
                    "name": name,
                    "algo": algo,
                    "fold_scores": [float(s) for s in scores] if scores else None,
                    "mean": _safe_float(mean),
                    "std": _safe_float(std),
                    "n": len(scores),
                }
            )

            if best_mean is None or mean > best_mean:
                best_mean = mean
                best_name = name

        best = {
            "name": best_name,
            "mean": _safe_float(best_mean) if best_mean is not None else None,
        }
        return per_est, best
