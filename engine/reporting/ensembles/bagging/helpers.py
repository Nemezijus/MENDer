from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def vote_margin_and_strength(preds_row: Sequence[Any]) -> Tuple[float, float, bool]:
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


def oob_coverage_from_decision_function(oob_decision: Any) -> Optional[float]:
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

__all__ = ["vote_margin_and_strength", "oob_coverage_from_decision_function"]
