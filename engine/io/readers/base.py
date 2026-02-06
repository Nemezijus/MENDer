from __future__ import annotations

"""Reader base contracts and shared helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Union

import numpy as np


@dataclass
class LoadedArray:
    """Parsed numeric array, optionally with feature names."""

    array: np.ndarray
    feature_names: Optional[list[str]] = None


class Reader(Protocol):
    """Protocol for parsing adapters."""

    def read(self, path: Union[str, Path], **kwargs) -> LoadedArray: ...


def coerce_numeric_matrix(arr: np.ndarray, *, context: str) -> np.ndarray:
    """Ensure a numeric, contiguous float array.

    - Coerces object dtype to float where possible
    - Rejects NaNs and non-numeric data
    """

    arr = np.asarray(arr)
    if arr.dtype == object:
        try:
            arr = arr.astype(float)
        except Exception as e:
            raise ValueError(f"{context}: could not convert to float; found non-numeric values.") from e

    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{context}: expected numeric data; got dtype={arr.dtype}")

    arr = np.asarray(arr, dtype=float)
    if np.isnan(arr).any():
        raise ValueError(f"{context}: contains NaN after parsing; check missing/invalid values.")

    return np.ascontiguousarray(arr)
