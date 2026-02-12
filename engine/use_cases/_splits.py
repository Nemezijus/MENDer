from __future__ import annotations

"""Splitter output normalization.

During refactoring, different splitter implementations existed and historically
returned different tuple shapes.

Use-cases should not duplicate heuristics for these shapes. This module provides
one helper for normalizing split outputs.
"""

from typing import Any

import numpy as np

from engine.components.splitters.types import Split


def unpack_split(split: Any) -> Split:
    """Normalize a splitter yield into :class:`engine.components.splitters.types.Split`.

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

    if isinstance(split, Split):
        return split

    if isinstance(split, (tuple, list)) and len(split) == 4:
        Xtr, Xte, ytr, yte = split
        return Split(Xtr=np.asarray(Xtr), Xte=np.asarray(Xte), ytr=np.asarray(ytr), yte=np.asarray(yte))

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
            return Split(
                Xtr=np.asarray(a2),
                Xte=np.asarray(a3),
                ytr=np.asarray(a4),
                yte=np.asarray(a5),
                idx_tr=np.asarray(a0, dtype=int).ravel(),
                idx_te=np.asarray(a1, dtype=int).ravel(),
            )

        # indices-last: (Xtr, Xte, ytr, yte, idx_tr, idx_te)
        if _looks_like_indices(a4) and _looks_like_indices(a5):
            return Split(
                Xtr=np.asarray(a0),
                Xte=np.asarray(a1),
                ytr=np.asarray(a2),
                yte=np.asarray(a3),
                idx_tr=np.asarray(a4, dtype=int).ravel(),
                idx_te=np.asarray(a5, dtype=int).ravel(),
            )

        # fallback: treat as (Xtr, Xte, ytr, yte, idx_tr, idx_te) but indices absent
        return Split(Xtr=np.asarray(a0), Xte=np.asarray(a1), ytr=np.asarray(a2), yte=np.asarray(a3))

    raise ValueError(
        "Splitter yielded an unsupported split tuple; expected 4 or 6 items. "
        f"Got {type(split).__name__} of length {len(split) if isinstance(split, (tuple, list)) else 'n/a'}."
    )
