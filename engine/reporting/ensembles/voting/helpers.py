from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def vote_margin_and_strength(
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

__all__ = ["vote_margin_and_strength"]
