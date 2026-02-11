from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np


def _hist_init(edges: np.ndarray) -> np.ndarray:
    return np.zeros(len(edges) - 1, dtype=float)


def _hist_add(counts: np.ndarray, values: Sequence[float], edges: np.ndarray) -> None:
    h, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    counts += h


def _effective_n_from_weights(w: np.ndarray) -> float:
    """Effective number of estimators (a.k.a. ESS) from weights.
    If weights are uniform -> ESS ~= n. If concentrated -> ESS smaller.
    """
    w = np.asarray(w, dtype=float)
    w = w[w > 0]
    if w.size == 0:
        return 0.0
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if s2 <= 0:
        return 0.0
    return (s1 * s1) / s2


def _weighted_margin_strength_tie(
    preds_row: Sequence[Any],
    weights: np.ndarray,
) -> Tuple[float, float, bool]:
    """Compute normalized weighted vote margin and strength for a single sample.

    - strength: top_weight / total_weight  (0..1)
    - margin: (top_weight - second_weight) / total_weight  (0..1)
    - tie: True if top_weight == second_weight (within tolerance)

    preds_row: predictions from each estimator for the sample
    weights: estimator weights (length m)
    """
    w = np.asarray(weights, dtype=float)
    m = len(preds_row)
    if w.size != m:
        # fallback: treat as uniform if mismatch
        w = np.ones(m, dtype=float)

    total = float(np.sum(w))
    if total <= 0:
        return 0.0, 0.0, True

    vote_w: Dict[Any, float] = {}
    for p, ww in zip(preds_row, w):
        vote_w[p] = vote_w.get(p, 0.0) + float(ww)

    vals = sorted(vote_w.values(), reverse=True)
    top = float(vals[0]) if vals else 0.0
    second = float(vals[1]) if len(vals) > 1 else 0.0

    strength = top / total
    margin = max(0.0, top - second) / total
    tie = bool(np.isclose(top, second))

    return float(margin), float(strength), tie

__all__ = [
    "_hist_init",
    "_hist_add",
    "_effective_n_from_weights",
    "_weighted_margin_strength_tie",
]
