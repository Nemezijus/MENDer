"""Parsing adapters (readers).

Readers are responsible for *format parsing* only (MAT/NPY/NPZ/CSV/XLSX/HDF5).
Orientation/shape coercion lives in :mod:`engine.core.shapes`.

"""

from .base import LoadedArray, Reader

from .mat_reader import MatReader, load_mat_variable
from .npy_reader import NpyReader, load_npy_array
from .npz_reader import NpzReader, load_npz_array
from .tabular_reader import TabularReader, load_delimited_table
from .xlsx_reader import XlsxReader, load_xlsx_table
from .hdf5_reader import Hdf5Reader, load_hdf5_dataset
from .auto_reader import AutoReader, read_array_auto, load_from_data_model

__all__ = [
    "LoadedArray",
    "Reader",
    "MatReader",
    "load_mat_variable",
    "NpyReader",
    "load_npy_array",
    "NpzReader",
    "load_npz_array",
    "TabularReader",
    "load_delimited_table",
    "XlsxReader",
    "load_xlsx_table",
    "Hdf5Reader",
    "load_hdf5_dataset",
    "AutoReader",
    "read_array_auto",
    "load_from_data_model",
]
