from __future__ import annotations

"""Auto-dispatching reader (by extension).

This module is responsible only for selecting the correct format reader based on
file extension and calling it with the appropriate parameters.

"""

from pathlib import Path
from typing import Optional, Union

from ..base import LoadedArray
from ..hdf5_reader import Hdf5Reader
from ..mat_reader import MatReader
from ..npy_reader import NpyReader
from ..npz_reader import NpzReader
from ..tabular_reader import TabularReader
from ..xlsx_reader import XlsxReader


def read_array_auto(
    path: Union[str, Path],
    *,
    # NPZ
    npz_key: Optional[str] = None,
    # tabular
    delimiter: Optional[str] = None,
    has_header: Optional[bool] = None,
    encoding: Optional[str] = None,
    # xlsx
    sheet_name: Optional[Union[str, int]] = None,
    # hdf5
    dataset_key: Optional[str] = None,
) -> LoadedArray:
    """Read an array from ``path`` based on file extension."""

    p = Path(path)
    suf = p.suffix.lower()

    if suf == ".mat":
        return MatReader().read(p)
    if suf == ".npy":
        return NpyReader().read(p)
    if suf == ".npz":
        key = npz_key or "X"
        return NpzReader(key=key).read(p)
    if suf in {".csv", ".tsv", ".txt"}:
        return TabularReader(delimiter=delimiter, has_header=has_header, encoding=encoding).read(p)
    if suf == ".xlsx":
        return XlsxReader(sheet_name=sheet_name, has_header=has_header).read(p)
    if suf in {".h5", ".hdf5"}:
        return Hdf5Reader(dataset_key=dataset_key).read(p)

    raise ValueError(
        f"Unsupported file extension '{p.suffix}' for path: {p}. "
        "Supported: .mat, .npy, .npz, .csv/.tsv/.txt, .xlsx, .h5/.hdf5"
    )
