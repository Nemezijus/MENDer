from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from engine.reporting.ensembles.common import vote_margin_and_strength


def update_all_agree_and_pairwise(
    *,
    base_preds: np.ndarray,
    pairwise_same: np.ndarray,
) -> Tuple[int, int]:
    """Update pairwise agreement counts in-place.

    Parameters
    ----------
    base_preds:
        (n, m) matrix of base estimator predictions.
    pairwise_same:
        (m, m) matrix holding *counts* of equal predictions.

    Returns
    -------
    all_agree_count, n
    """
    P = np.asarray(base_preds)
    if P.ndim != 2:
        return 0, 0

    n = int(P.shape[0])
    m = int(P.shape[1])
    if m == 0 or n == 0:
        return 0, n

    all_agree = np.all(P == P[:, [0]], axis=1)
    all_agree_count = int(np.sum(all_agree))

    # Count equalities per (i, j)
    for i in range(m):
        for j in range(i, m):
            same = float(np.sum(P[:, i] == P[:, j]))
            pairwise_same[i, j] += same
            pairwise_same[j, i] += same if i != j else 0.0

    return all_agree_count, n


def margins_strengths_and_ties(
    *,
    base_preds: np.ndarray,
    weights: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute vote margin/strength per sample, and tie count."""
    P = np.asarray(base_preds)
    if P.ndim != 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), 0

    n = int(P.shape[0])
    margins = np.zeros(n, dtype=float)
    strengths = np.zeros(n, dtype=float)
    ties = 0

    for r in range(n):
        margin, strength, tie = vote_margin_and_strength(P[r, :], weights=weights)
        margins[r] = float(margin)
        strengths[r] = float(strength)
        if tie:
            ties += 1
    return margins, strengths, ties


def hist_add_inplace(*, counts: np.ndarray, edges: np.ndarray, values: np.ndarray) -> None:
    """Update histogram counts in-place."""
    if counts is None or edges is None:
        return
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return
    h, _ = np.histogram(vals, bins=edges)
    counts += h


def finalize_pairwise_agreement(
    *,
    pairwise_same: np.ndarray,
    n_total: int,
) -> Tuple[Optional[float], Optional[List[List[float]]]]:
    """Return (pairwise_mean_agreement, matrix_as_list) for report output."""
    denom = float(n_total) if n_total > 0 else 1.0
    if pairwise_same is None:
        return None, None
    pairwise = np.asarray(pairwise_same, dtype=float) / denom
    if pairwise.size == 0:
        return None, []

    m = pairwise.shape[0]
    if m > 1:
        mask = ~np.eye(m, dtype=bool)
        pairwise_mean = float(np.mean(pairwise[mask]))
    else:
        pairwise_mean = 1.0
    return pairwise_mean, pairwise.tolist()
