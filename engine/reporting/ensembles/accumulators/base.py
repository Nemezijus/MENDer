from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from engine.reporting.ensembles.common import _mean_std, _safe_float


@dataclass
class FoldAccumulatorBase:
    """Fold accumulator base that tracks an internal fold counter."""

    _n_folds: int = field(default=0, init=False)

    def _bump_fold(self) -> None:
        self._n_folds += 1


class PerEstimatorFoldAccumulatorBase(FoldAccumulatorBase):
    """Mixin/base with helpers for per-estimator fold score bookkeeping.

    This is intentionally NOT a dataclass to avoid dataclass field-order issues
    in subclasses that have required (non-default) fields.
    """

    _scores: Dict[str, List[float]] | None

    def _init_scores(self) -> None:
        names = list(getattr(self, 'estimator_names', None) or [])
        self._scores = {n: [] for n in names}

    def _record_fold_scores(self, base_scores: Dict[str, float]) -> None:
        if not base_scores or not getattr(self, '_scores', None):
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

        estimator_names: Sequence[str] = getattr(self, 'estimator_names', None) or []
        estimator_algos: Sequence[str] = getattr(self, 'estimator_algos', None) or []

        for name, algo in zip(estimator_names, estimator_algos):
            scores = (getattr(self, '_scores', None) or {}).get(name, [])
            mean, std = _mean_std(scores)
            per_est.append(
                {
                    'name': name,
                    'algo': algo,
                    'fold_scores': [float(s) for s in scores] if scores else None,
                    'mean': _safe_float(mean),
                    'std': _safe_float(std),
                    'n': len(scores),
                }
            )

            if best_mean is None or mean > best_mean:
                best_mean = mean
                best_name = name

        best = {
            'name': best_name,
            'mean': _safe_float(best_mean) if best_mean is not None else None,
        }
        return per_est, best
