from __future__ import annotations

"""Splitter output normalization.

During refactoring, different splitter implementations existed and historically
returned different tuple shapes.

Use-cases should not duplicate heuristics for these shapes. This module provides
one helper for normalizing split outputs.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class NormalizedSplit:
    """Canonical split container."""

    Xtr: Any
    Xte: Any
    ytr: Any
    yte: Any
    idx_tr: Optional[np.ndarray]
    idx_te: Optional[np.ndarray]


def unpack_split(split: Any) -> NormalizedSplit:
    """Normalize splitter yields into :class:`NormalizedSplit`.

    Supported formats:
      - (Xtr, Xte, ytr, yte)
      - (idx_tr, idx_te, Xtr, Xte, ytr, yte)
      - (Xtr, Xte, ytr, yte, idx_tr, idx_te)
      - (Xtr, Xte, ytr, yte, idx_tr, idx_te) where idx_tr/idx_te may be None

    Notes
    -----
    Only ``idx_tr``/``idx_te`` are optional; the arrays themselves are passed
    through as-is and may be numpy arrays, pandas objects, etc.
    """

    if isinstance(split, (tuple, list)) and len(split) == 4:
        Xtr, Xte, ytr, yte = split
        return NormalizedSplit(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, idx_tr=None, idx_te=None)

    if isinstance(split, (tuple, list)) and len(split) == 6:
        a0, a1, a2, a3, a4, a5 = split

        def _looks_like_indices(x: Any) -> bool:
            if x is None:
                return False
            try:
                arr = np.asarray(x)
                return arr.ndim == 1 and arr.size >= 1 and np.issubdtype(arr.dtype, np.integer)
            except Exception:
                return False

        # indices-first: (idx_tr, idx_te, Xtr, Xte, ytr, yte)
        if _looks_like_indices(a0) and _looks_like_indices(a1):
            return NormalizedSplit(
                Xtr=a2,
                Xte=a3,
                ytr=a4,
                yte=a5,
                idx_tr=np.asarray(a0, dtype=int).ravel(),
                idx_te=np.asarray(a1, dtype=int).ravel(),
            )

        # indices-last: (Xtr, Xte, ytr, yte, idx_tr, idx_te)
        if _looks_like_indices(a4) and _looks_like_indices(a5):
            return NormalizedSplit(
                Xtr=a0,
                Xte=a1,
                ytr=a2,
                yte=a3,
                idx_tr=np.asarray(a4, dtype=int).ravel(),
                idx_te=np.asarray(a5, dtype=int).ravel(),
            )

        # fallback: treat as (Xtr, Xte, ytr, yte, idx_tr, idx_te) but indices absent
        return NormalizedSplit(Xtr=a0, Xte=a1, ytr=a2, yte=a3, idx_tr=None, idx_te=None)

    raise ValueError(
        "Splitter yielded an unsupported split tuple; expected 4 or 6 items. "
        f"Got {type(split).__name__} of length {len(split) if isinstance(split, (tuple, list)) else 'n/a'}."
    )
