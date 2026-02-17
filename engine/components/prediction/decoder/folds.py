from __future__ import annotations

"""Fold-id helpers for decoder outputs."""

from typing import Iterable, List, Optional

import numpy as np


def parse_fold_ids(fold_ids: Optional[Iterable[int]], n: int) -> Optional[List[int]]:
    """Parse fold IDs into a validated list of length n."""

    if fold_ids is None:
        return None
    try:
        out = [int(v) for v in list(fold_ids)]
    except Exception:
        return None
    return out if len(out) == int(n) else None


def ensure_holdout_fold_ids(*, mode: str, fold_ids: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    """Legacy behavior: for holdout, set fold_id=1 if missing."""

    if fold_ids is not None:
        return fold_ids
    if str(mode).lower() != "holdout":
        return None
    try:
        return np.ones((int(n),), dtype=int)
    except Exception:
        return None
