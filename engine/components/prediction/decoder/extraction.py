from __future__ import annotations

"""Decoder output extraction helpers.

Raw extraction (decision_function / predict_proba etc.) is implemented in
``engine.components.prediction.decoder_extraction``.

This module contains small helper computations performed on already-extracted
arrays (e.g. margin computation).
"""

from typing import Optional

import numpy as np


def compute_margin(
    *,
    decision_scores: Optional[np.ndarray],
    proba: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Best-effort confidence proxy.

    - Prefer decision scores when present.
    - Fall back to probabilities/vote shares.
    """

    if decision_scores is not None:
        ds = np.asarray(decision_scores)
        if ds.ndim == 1:
            return np.abs(ds)
        if ds.ndim == 2 and ds.shape[1] >= 2:
            part = np.partition(ds, kth=-2, axis=1)
            top2 = part[:, -2]
            top1 = part[:, -1]
            return top1 - top2
        return None

    if proba is not None:
        p = np.asarray(proba)
        if p.ndim == 2 and p.shape[1] >= 2:
            if p.shape[1] == 2:
                return np.abs(p[:, 1] - p[:, 0])
            part = np.partition(p, kth=-2, axis=1)
            return part[:, -1] - part[:, -2]
    return None
