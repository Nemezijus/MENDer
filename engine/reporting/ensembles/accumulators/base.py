from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FoldAccumulatorBase:
    """Minimal shared base for fold-based accumulators.

    The goal is not to be clever — just to standardize the most common book-keeping
    fields so accumulator implementations stay consistent.
    """

    _n_folds: int = 0

    def _bump_fold(self) -> None:
        self._n_folds += 1
