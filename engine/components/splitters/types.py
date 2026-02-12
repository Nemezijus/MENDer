from __future__ import annotations

"""Splitter return contracts.

The splitters in MENDer are required to yield a *single, stable* fold payload
shape. This avoids tuple-shape guessing in orchestrators and makes downstream
typing and testing significantly simpler.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Split:
    """A single train/test split (fold).

    Notes
    -----
    - `idx_tr` / `idx_te` are optional row indices into the *original* X/y.
      They are helpful for re-ordering pooled outputs and for UI traceability.
    - For holdout splitters, indices may be omitted (None).
    """

    Xtr: np.ndarray
    Xte: np.ndarray
    ytr: np.ndarray
    yte: np.ndarray
    idx_tr: Optional[np.ndarray] = None
    idx_te: Optional[np.ndarray] = None
