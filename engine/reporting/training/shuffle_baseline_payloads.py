from __future__ import annotations

"""Reporting helpers for the label-shuffle baseline.

The label-shuffle baseline produces a distribution of evaluation scores under
random label permutations. We summarize it in a JSON-friendly shape compatible
with the public TrainResult/EnsembleResult contracts.

Historically, some of this logic lived in the backend service layer. Segment 12
moves it into BL reporting so both engine use-cases and backend adapters share
one implementation.

Contract policy
---------------
The returned dicts must only include keys that are valid in the training result
contracts (extra fields are forbidden at the boundary).
"""

from typing import Any, Dict, Sequence

import numpy as np

from engine.reporting.common.json_safety import safe_float_list


def build_label_shuffle_baseline_section(*, scores: Sequence[float], ref_score: Any) -> Dict[str, Any]:
    """Build the baseline section to be merged into a training payload.

    Parameters
    ----------
    scores:
        Sequence of baseline scores (one per shuffle).
    ref_score:
        Reference score to compare against (typically the trained model's mean score).

    Returns
    -------
    dict
        Keys are a strict subset of the TrainResult contract:
        - shuffled_scores: List[float]
        - p_value: float
        - notes: List[str] (caller is expected to extend existing notes)
    """

    # Ensure the score distribution is JSON-safe (no NaN/inf) before emitting.
    scores_safe = np.asarray(safe_float_list(scores), dtype=float).ravel()

    try:
        ref_f = float(ref_score)
    except Exception:
        ref_f = float("nan")

    n = int(scores_safe.size)
    if n < 1:
        p_val = float("nan")
        mean = float("nan")
        std = float("nan")
        shuffled = []
    else:
        ge = int(np.sum(scores_safe >= ref_f))
        p_val = (ge + 1.0) / (n + 1.0)
        mean = float(np.mean(scores_safe))
        std = float(np.std(scores_safe))
        shuffled = [float(v) for v in scores_safe.tolist()]

    note = f"Shuffle baseline: mean={mean:.4f} ± {std:.4f}, p≈{float(p_val):.4f}"

    return {
        "shuffled_scores": shuffled,
        "p_value": float(p_val),
        "notes": [note],
    }


def format_label_shuffle_baseline_failure_note(
    *,
    exc_type: str,
    exc_message: str,
    parens: bool = False,
) -> str:
    """Create a user-facing note when baseline execution fails."""

    if parens:
        return f"Shuffle baseline failed ({exc_type}: {exc_message})."
    return f"Shuffle baseline failed: {exc_type}: {exc_message}"
