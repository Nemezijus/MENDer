from __future__ import annotations

"""MATLAB .mat reader."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from scipy.io import loadmat as scipy_loadmat

from .base import LoadedArray


def load_mat_variable(
    file_path: Union[str, Path],
    *,
    squeeze: bool = True,
    allow_hdf5_fallback: bool = True,
) -> np.ndarray:
    """Load exactly one numeric variable from a MATLAB .mat file."""

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"MAT-file not found: {file_path}")

    try:
        mat = scipy_loadmat(
            file_path.as_posix(),
            squeeze_me=squeeze,
            struct_as_record=False,
        )
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if len(keys) == 0:
            raise ValueError("No user variables found in MAT-file.")
        if len(keys) > 1:
            raise ValueError(
                f"Expected exactly one variable, found {len(keys)}: {keys}. "
                "Please save a MAT file with a single variable."
            )

        name = keys[0]
        arr = np.asarray(mat[name])
        if squeeze:
            arr = np.squeeze(arr)

        if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
            raise TypeError(
                f"Variable '{name}' is not a plain numeric array (dtype={arr.dtype}). "
                "Structs/cells/tables are not supported — save the underlying numeric field/array."
            )

        return np.ascontiguousarray(arr)

    except (NotImplementedError, OSError) as e:
        if not allow_hdf5_fallback:
            raise RuntimeError(
                "This MAT-file appears to be MATLAB v7.3 (HDF5). "
                "Install 'h5py' or re-save the file as v7 or earlier."
            ) from e

        try:
            import h5py  # type: ignore
        except Exception as ee:
            raise RuntimeError(
                "Reading MATLAB v7.3 (.mat) requires the 'h5py' package. "
                "Install it (e.g., `pip install h5py`) or re-save the file as v7 or earlier."
            ) from ee

        with h5py.File(file_path.as_posix(), "r") as f:
            keys = [k for k in f.keys() if not k.startswith("#")]
            if len(keys) == 0:
                raise ValueError("No user variables found in v7.3 MAT-file.")
            if len(keys) > 1:
                raise ValueError(
                    f"Expected exactly one variable in v7.3 MAT-file, found {len(keys)}: {keys}. "
                    "Please save a MAT file with a single variable."
                )

            name = keys[0]
            node = f[name]
            if not hasattr(node, "shape"):
                raise TypeError(
                    f"Variable '{name}' is not a numeric dataset (found a group). "
                    "Structs/cells/tables are not supported — save the numeric array directly."
                )

            arr = np.array(node)
            if squeeze:
                arr = np.squeeze(arr)
            if not np.issubdtype(arr.dtype, np.number):
                raise TypeError(
                    f"Variable '{name}' is not numeric (dtype={arr.dtype}). "
                    "Structs/cells/tables are not supported."
                )
            return np.ascontiguousarray(arr)


@dataclass
class MatReader:
    squeeze: bool = True
    allow_hdf5_fallback: bool = True

    def read(self, path: Union[str, Path], **kwargs) -> LoadedArray:
        arr = load_mat_variable(
            path,
            squeeze=bool(kwargs.get("squeeze", self.squeeze)),
            allow_hdf5_fallback=bool(kwargs.get("allow_hdf5_fallback", self.allow_hdf5_fallback)),
        )
        return LoadedArray(array=np.asarray(arr), feature_names=None)
