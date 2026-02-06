from __future__ import annotations

"""HDF5 .h5/.hdf5 reader."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from .base import LoadedArray, coerce_numeric_matrix


def _iter_datasets(h5file) -> Iterable[str]:
    def _walk(group, prefix: str = ""):
        for k, v in group.items():
            p = f"{prefix}/{k}" if prefix else k
            if k.startswith("#"):
                continue
            if hasattr(v, "shape"):
                yield p
            else:
                yield from _walk(v, p)
    return _walk(h5file)


def load_hdf5_dataset(
    file_path: Union[str, Path],
    *,
    dataset_key: Optional[str] = None,
) -> LoadedArray:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    try:
        import h5py  # type: ignore
    except Exception as e:
        raise RuntimeError("Loading .h5/.hdf5 requires h5py. Install it (pip install h5py).") from e

    with h5py.File(path.as_posix(), "r") as f:
        key = dataset_key
        if key is None:
            ds_paths = list(_iter_datasets(f))
            if not ds_paths:
                raise ValueError(f"{path.name}: no datasets found in HDF5 file")
            key = ds_paths[0]

        if key not in f:
            ds_paths = list(_iter_datasets(f))
            raise KeyError(
                f"Dataset '{key}' not found in {path.name}. "
                f"Available datasets: {ds_paths[:30]}" + (" ..." if len(ds_paths) > 30 else "")
            )

        node = f[key]
        arr = np.array(node)
        arr = coerce_numeric_matrix(arr, context=f"HDF5 '{path.name}:{key}'")
        return LoadedArray(arr, None)


@dataclass
class Hdf5Reader:
    dataset_key: Optional[str] = None

    def read(self, path: Union[str, Path], **kwargs) -> LoadedArray:
        return load_hdf5_dataset(
            path,
            dataset_key=kwargs.get("dataset_key", self.dataset_key),
        )
