from __future__ import annotations

"""NumPy .npz reader."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .base import LoadedArray


def load_npz_array(file_path: Union[str, Path], *, key: str) -> LoadedArray:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")
    data = np.load(path.as_posix(), allow_pickle=True)
    if key not in data.files:
        raise KeyError(f"Key '{key}' not found in {path.name}. Available: {list(data.keys())}")
    arr = np.asarray(data[key])
    return LoadedArray(array=np.ascontiguousarray(arr), feature_names=None)


@dataclass
class NpzReader:
    key: str = "X"

    def read(self, path: Union[str, Path], **kwargs) -> LoadedArray:
        key = str(kwargs.get("key", self.key))
        return load_npz_array(path, key=key)
