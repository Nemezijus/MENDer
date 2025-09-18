from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat as scipy_loadmat

def load_mat_variable(
    file_path: str | Path,
    *,
    squeeze: bool = True,
    allow_hdf5_fallback: bool = True,
) -> np.ndarray:
    """
    Load exactly one variable from a MATLAB .mat file and return it as a NumPy array.

    Assumptions:
      - The .mat file contains *one* user variable (not a whole workspace dump).
      - The variable is a plain numeric array (NOT a struct, table, or cell).
    Basic checks:
      - Errors if no variables or more than one are present.
      - Errors if the variable is non-numeric (struct/cell/table).
      - Graceful message for MATLAB v7.3 files (HDF5) with optional h5py fallback.

    Parameters
    ----------
    file_path : str | Path
        Path to the .mat file.
    squeeze : bool
        If True, squeeze singleton dimensions (default: True).
    allow_hdf5_fallback : bool
        If True and the file is MATLAB v7.3, use a light h5py fallback when available.

    Returns
    -------
    np.ndarray
        The loaded numeric array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If there are zero or multiple variables in the file.
    TypeError
        If the variable is not a plain numeric array (e.g., struct/cell/table).
    RuntimeError
        If the file is MATLAB v7.3 but h5py is not installed (and fallback is allowed).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"MAT-file not found: {file_path}")

    # Try classic MAT (v5/v7) with SciPy first.
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

        # Reject non-numeric arrays (structs/cells/tables often come through as object dtype)
        if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
            raise TypeError(
                f"Variable '{name}' is not a plain numeric array (dtype={arr.dtype}). "
                "Structs/cells/tables are not supported — save the underlying numeric field/array."
            )

        return np.ascontiguousarray(arr)

    except (NotImplementedError, OSError) as e:
        # Likely a MATLAB v7.3 (HDF5-based) file
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

        # Minimal HDF5 fallback: expect exactly one top-level dataset
        with h5py.File(file_path.as_posix(), "r") as f:
            # Ignore MATLAB-internal groups like '#refs#'
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
            # For simple arrays, we expect a Dataset. Groups usually indicate structs/tables.
            if not isinstance(node, h5py.Dataset):
                raise TypeError(
                    f"Variable '{name}' is not a numeric dataset (found a group). "
                    "Structs/cells/tables are not supported — save the numeric array directly."
                )

            arr = np.array(node)  # load into memory
            if squeeze:
                arr = np.squeeze(arr)

            if not np.issubdtype(arr.dtype, np.number):
                raise TypeError(
                    f"Variable '{name}' is not numeric (dtype={arr.dtype}). "
                    "Structs/cells/tables are not supported."
                )

            return np.ascontiguousarray(arr)
        
if __name__ == "__main__":
    import sys
    try:
        a = load_mat_variable(sys.argv[1] if len(sys.argv) > 1 else "example.mat")
        print("Loaded array:", a.shape, a.dtype)
    except Exception as err:
        print("Error:", err)