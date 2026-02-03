from __future__ import annotations

"""Low-level file readers for MENDer.

Design goals
------------
- Keep *format parsing* here (MAT/NPY/CSV/XLSX/HDF5) so loaders stay thin.
- For tabular formats, optionally preserve feature names from a header row.
- Return NumPy arrays suitable for modelling; callers handle orientation/coercion.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union, Iterable

import numpy as np
from scipy.io import loadmat as scipy_loadmat


@dataclass
class LoadedArray:
    """Return type for loaders that may carry feature names."""

    array: np.ndarray
    feature_names: Optional[list[str]] = None


# ------------------------------ helpers ---------------------------------


def _read_text_head(path: Path, n_lines: int = 5, encoding: Optional[str] = None) -> list[str]:
    enc = encoding or "utf-8"
    lines: list[str] = []
    with path.open("r", encoding=enc, errors="replace") as f:
        for _ in range(n_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line.strip("\n"))
    return lines


def _infer_delimiter(sample_line: str) -> str:
    # Prefer common explicit delimiters.
    if "\t" in sample_line:
        return "\t"
    if "," in sample_line:
        return ","
    if ";" in sample_line:
        return ";"
    # Fallback: whitespace-separated.
    return "whitespace"


def _split_line(line: str, delimiter: str) -> list[str]:
    if delimiter == "whitespace":
        return [t for t in line.strip().split() if t != ""]
    return [t.strip() for t in line.split(delimiter)]


def _row_is_numeric(tokens: Sequence[str]) -> bool:
    if len(tokens) == 0:
        return False
    try:
        for t in tokens:
            if t == "":
                return False
            float(t)
        return True
    except Exception:
        return False


def _coerce_numeric_matrix(arr: np.ndarray, *, context: str) -> np.ndarray:
    arr = np.asarray(arr)
    # If pandas gave us object dtype, try to coerce.
    if arr.dtype == object:
        try:
            arr = arr.astype(float)
        except Exception as e:
            raise ValueError(f"{context}: could not convert to float; found non-numeric values.") from e
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{context}: expected numeric data; got dtype={arr.dtype}")
    # Replace inf with nan so downstream can catch.
    arr = np.asarray(arr, dtype=float)
    if np.isnan(arr).any():
        raise ValueError(f"{context}: contains NaN after parsing; check missing/invalid values.")
    return np.ascontiguousarray(arr)


# ------------------------------ .mat ------------------------------------


def load_mat_variable(
    file_path: str | Path,
    *,
    squeeze: bool = True,
    allow_hdf5_fallback: bool = True,
) -> np.ndarray:
    """Load exactly one variable from a MATLAB .mat file and return it as a NumPy array.

    Assumptions:
      - The .mat file contains *one* user variable (not a whole workspace dump).
      - The variable is a plain numeric array (NOT a struct, table, or cell).
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


# ------------------------------ .npy ------------------------------------


def load_npy_array(file_path: str | Path) -> LoadedArray:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"NPY file not found: {path}")
    arr = np.load(path.as_posix(), allow_pickle=False)
    arr = _coerce_numeric_matrix(arr, context=f"NPY '{path.name}'")
    return LoadedArray(arr, None)


# ------------------------------ delimited text --------------------------


def load_delimited_table(
    file_path: str | Path,
    *,
    delimiter: Optional[str] = None,
    has_header: Optional[bool] = None,
    encoding: Optional[str] = None,
) -> LoadedArray:
    """Load CSV/TSV/TXT numeric table.

    - If has_header is None, header is inferred by checking whether the first row is numeric.
    - If delimiter is None, it is inferred from the first line.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Table file not found: {path}")

    head_lines = _read_text_head(path, n_lines=2, encoding=encoding)
    first = head_lines[0] if head_lines else ""

    delim = delimiter
    if delim is None:
        delim = _infer_delimiter(first)
    if delim == "\\t":
        delim = "\t"

    tokens = _split_line(first, delim)
    inferred_header = not _row_is_numeric(tokens)
    use_header = inferred_header if has_header is None else bool(has_header)

    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Loading CSV/TSV/TXT requires pandas. Install it (pip install pandas)."
        ) from e

    # sep handling
    read_kwargs = dict(
        encoding=encoding or "utf-8",
        engine="python",
    )
    if delim == "whitespace":
        # regex separator for arbitrary whitespace
        sep = r"\s+"
    else:
        sep = delim

    header_arg = 0 if use_header else None
    df = pd.read_csv(path.as_posix(), sep=sep, header=header_arg, **read_kwargs)

    feature_names: Optional[list[str]] = None
    if use_header:
        feature_names = [str(c) for c in df.columns.tolist()]

    # Coerce to numeric
    df_num = df.apply(pd.to_numeric, errors="coerce")
    if df_num.isna().to_numpy().any():
        # Provide a readable error message.
        n_bad = int(df_num.isna().to_numpy().sum())
        raise ValueError(
            f"{path.name}: found {n_bad} non-numeric/missing cells after parsing. "
            "Clean the file or export as purely numeric values."
        )

    arr = _coerce_numeric_matrix(df_num.to_numpy(), context=f"Table '{path.name}'")
    return LoadedArray(arr, feature_names)


# ------------------------------ .xlsx -----------------------------------


def load_xlsx_table(
    file_path: str | Path,
    *,
    sheet_name: Optional[Union[str, int]] = None,
    has_header: Optional[bool] = None,
) -> LoadedArray:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"XLSX file not found: {path}")

    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("Loading XLSX requires pandas. Install it (pip install pandas).") from e

    hdr = 0 if (has_header is True) else None
    df = pd.read_excel(path.as_posix(), sheet_name=sheet_name or 0, header=hdr)

    feature_names: Optional[list[str]] = None
    if has_header is True:
        feature_names = [str(c) for c in df.columns.tolist()]
    elif has_header is None:
        # infer header by checking whether column names are default integers
        # if pandas read without header, columns are 0..n-1
        feature_names = None

    df_num = df.apply(pd.to_numeric, errors="coerce")
    if df_num.isna().to_numpy().any():
        n_bad = int(df_num.isna().to_numpy().sum())
        raise ValueError(
            f"{path.name}: found {n_bad} non-numeric/missing cells after parsing. "
            "Clean the sheet or export as numeric-only values."
        )
    arr = _coerce_numeric_matrix(df_num.to_numpy(), context=f"XLSX '{path.name}'")
    return LoadedArray(arr, feature_names)


# ------------------------------ .h5/.hdf5 --------------------------------


def _iter_datasets(h5file) -> Iterable[str]:
    # depth-first traversal returning dataset paths
    def _walk(group, prefix: str = ""):
        for k, v in group.items():
            p = f"{prefix}/{k}" if prefix else k
            # skip matlab refs
            if k.startswith("#"):
                continue
            if hasattr(v, "shape"):
                yield p
            else:
                yield from _walk(v, p)

    return _walk(h5file)


def load_hdf5_dataset(
    file_path: str | Path,
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
            # pick the first dataset we can find
            ds_paths = list(_iter_datasets(f))
            if not ds_paths:
                raise ValueError(f"{path.name}: no datasets found in HDF5 file")
            key = ds_paths[0]

        if key not in f:
            # Provide helpful message with available datasets
            ds_paths = list(_iter_datasets(f))
            raise KeyError(
                f"Dataset '{key}' not found in {path.name}. "
                f"Available datasets: {ds_paths[:30]}" + (" ..." if len(ds_paths) > 30 else "")
            )

        node = f[key]
        arr = np.array(node)
        arr = _coerce_numeric_matrix(arr, context=f"HDF5 '{path.name}:{key}'")
        return LoadedArray(arr, None)


if __name__ == "__main__":
    # minimal manual sanity
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else None
    if not p:
        print("Usage: python data_read.py <path>")
        raise SystemExit(1)

    path = Path(p)
    suf = path.suffix.lower()
    if suf == ".mat":
        a = load_mat_variable(path)
        print("MAT", a.shape, a.dtype)
    elif suf == ".npy":
        a = load_npy_array(path)
        print("NPY", a.array.shape, a.array.dtype)
    elif suf in {".csv", ".tsv", ".txt"}:
        a = load_delimited_table(path)
        print("TAB", a.array.shape, a.feature_names[:5] if a.feature_names else None)
    elif suf in {".h5", ".hdf5"}:
        a = load_hdf5_dataset(path)
        print("H5", a.array.shape)
    elif suf == ".xlsx":
        a = load_xlsx_table(path)
        print("XLSX", a.array.shape)
    else:
        print("Unsupported")
