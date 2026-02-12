from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from engine.reporting.common.json_safety import ReportError, add_report_error


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
        try:
            w = float(w)
        except Exception:
            w = 0.0
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


def _safe_float(x: Any) -> float:
    try:
        return float(np.nan_to_num(float(x), nan=0.0, posinf=1.0, neginf=0.0))
    except Exception:
        return 0.0


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    a = np.asarray(xs, dtype=float)
    return float(np.mean(a)), float(np.std(a))


def attach_report_error(target: Any, *, where: str, exc: BaseException, context: dict[str, Any] | None = None) -> None:
    """Attach a structured error marker to an accumulator/payload object."""
    try:
        errors = getattr(target, "_errors", None)
        if errors is None:
            errors = []
            setattr(target, "_errors", errors)
        add_report_error(errors, where=where, exc=exc, context=context)
    except Exception:
        pass
