from __future__ import annotations

"""NumPy .npy reader."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from .base import LoadedArray, coerce_numeric_matrix


def load_npy_array(file_path: Union[str, Path]) -> LoadedArray:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"NPY file not found: {path}")
    arr = np.load(path.as_posix(), allow_pickle=False)
    arr = coerce_numeric_matrix(arr, context=f"NPY '{path.name}'")
    return LoadedArray(arr, None)


@dataclass
class NpyReader:
    def read(self, path: Union[str, Path], **kwargs) -> LoadedArray:
        return load_npy_array(path)
