from __future__ import annotations

"""Legacy low-level file readers.

NOTE
----
This module is kept for backwards compatibility only. The canonical implementation
now lives under :mod:`engine.io.readers`.

Prefer importing from ``engine.io.readers`` in new code.
"""

from engine.io.readers import (  # noqa: F401
    LoadedArray,
    load_mat_variable,
    load_npy_array,
    load_npz_array,
    load_delimited_table,
    load_xlsx_table,
    load_hdf5_dataset,
)

__all__ = [
    "LoadedArray",
    "load_mat_variable",
    "load_npy_array",
    "load_npz_array",
    "load_delimited_table",
    "load_xlsx_table",
    "load_hdf5_dataset",
]
