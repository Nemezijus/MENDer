from __future__ import annotations

"""Concatenation helpers for per-fold decoder outputs."""

from typing import List, Optional

import numpy as np

from .errors import DecoderOutputsConcatError


def concat_if_complete(
    parts: List[Optional[np.ndarray]],
    *,
    name: str,
    notes: List[str],
) -> Optional[np.ndarray]:
    """Concatenate parts if all folds produced the output.

    If at least one fold omitted the output (None), returns None and appends a note.
    """

    if not parts:
        return None
    if any(p is None for p in parts):
        notes.append(f"Decoder outputs: '{name}' omitted because at least one fold could not produce it.")
        return None
    try:
        return np.concatenate([np.asarray(p) for p in parts], axis=0)
    except Exception as e:
        raise DecoderOutputsConcatError(
            f"Cannot concatenate decoder outputs '{name}' ({type(e).__name__}: {e})"
        ) from e


def reorder_if_possible(arr: Optional[np.ndarray], order: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Reorder an array with an order mapping if shapes match."""

    if arr is None or order is None:
        return arr
    try:
        if int(arr.shape[0]) == int(order.shape[0]):
            return arr[order]
    except Exception:
        return arr
    return arr
