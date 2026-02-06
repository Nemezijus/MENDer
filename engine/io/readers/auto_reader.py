from __future__ import annotations

"""Auto-dispatching reader + DataModel convenience loader."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from engine.contracts.run_config import DataModel
from engine.core.shapes import align_X_y, coerce_X_only

from .base import LoadedArray
from .mat_reader import MatReader
from .npy_reader import NpyReader
from .npz_reader import NpzReader
from .tabular_reader import TabularReader
from .xlsx_reader import XlsxReader
from .hdf5_reader import Hdf5Reader


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


@dataclass
class AutoReader:
    """Stateful wrapper mainly for dependency injection."""

    def read(self, path: Union[str, Path], **kwargs) -> LoadedArray:
        return read_array_auto(path, **kwargs)


def load_from_data_model(cfg: DataModel) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[list[str]]]:
    """Load X and optional y using a :class:`~shared_schemas.run_config.DataModel`.

    Returns
    -------
    X : np.ndarray (2D)
    y : np.ndarray | None (1D)
    feature_names : list[str] | None
    """

    # Prefer explicit npz_path if provided
    if cfg.npz_path:
        npz_path = cfg.npz_path
        if not cfg.x_key:
            raise ValueError("NPZ loading requires x_key")
        X_loaded = read_array_auto(npz_path, npz_key=cfg.x_key)

        y: Optional[np.ndarray] = None
        if cfg.y_key:
            try:
                y_loaded = read_array_auto(npz_path, npz_key=cfg.y_key)
                y = np.asarray(y_loaded.array).ravel()
                X, y = align_X_y(X_loaded.array, y)
                return X, y, X_loaded.feature_names
            except KeyError:
                y = None

        X = coerce_X_only(X_loaded.array)
        return X, None, X_loaded.feature_names

    if not cfg.x_path:
        raise ValueError("DataModel requires x_path (or npz_path)")

    x_path = cfg.x_path
    x_lower = x_path.lower()

    # --- X ---------------------------------------------------------------
    if x_lower.endswith(".npz"):
        if not cfg.x_key:
            raise ValueError("NPZ loading requires x_key")
        X_loaded = read_array_auto(x_path, npz_key=cfg.x_key)
    elif x_lower.endswith((".h5", ".hdf5")):
        X_loaded = read_array_auto(x_path, dataset_key=cfg.x_key)
    elif x_lower.endswith((".csv", ".tsv", ".txt")):
        X_loaded = read_array_auto(
            x_path,
            delimiter=cfg.delimiter,
            has_header=cfg.has_header,
            encoding=cfg.encoding,
        )
    elif x_lower.endswith(".xlsx"):
        X_loaded = read_array_auto(
            x_path,
            sheet_name=cfg.sheet_name,
            has_header=cfg.has_header,
        )
    else:
        X_loaded = read_array_auto(x_path)

    # --- y ---------------------------------------------------------------
    y: Optional[np.ndarray] = None

    if cfg.y_path:
        y_path = cfg.y_path
        y_lower = y_path.lower()

        if y_lower.endswith(".npz"):
            if not cfg.y_key:
                raise ValueError("NPZ y_path requires y_key")
            y_loaded = read_array_auto(y_path, npz_key=cfg.y_key)
            y = np.asarray(y_loaded.array).ravel()
            X, y = align_X_y(X_loaded.array, y)
            return X, y, X_loaded.feature_names

        if y_lower.endswith((".h5", ".hdf5")):
            y_loaded = read_array_auto(y_path, dataset_key=cfg.y_key)
            y = np.asarray(y_loaded.array).ravel()
            X, y = align_X_y(X_loaded.array, y)
            return X, y, X_loaded.feature_names

        if y_lower.endswith((".csv", ".tsv", ".txt")):
            # y usually has no header; do not force cfg.has_header=True onto y.
            y_has_header = None if (cfg.has_header is True) else cfg.has_header
            y_loaded = read_array_auto(
                y_path,
                delimiter=cfg.delimiter,
                has_header=y_has_header,
                encoding=cfg.encoding,
            )
            y = np.asarray(y_loaded.array).ravel()
            X, y = align_X_y(X_loaded.array, y)
            return X, y, X_loaded.feature_names

        if y_lower.endswith(".xlsx"):
            y_has_header = None if (cfg.has_header is True) else cfg.has_header
            y_loaded = read_array_auto(
                y_path,
                sheet_name=cfg.sheet_name,
                has_header=y_has_header,
            )
            y = np.asarray(y_loaded.array).ravel()
            X, y = align_X_y(X_loaded.array, y)
            return X, y, X_loaded.feature_names

        if y_lower.endswith(".mat") or y_lower.endswith(".npy"):
            y_loaded = read_array_auto(y_path)
            y = np.asarray(y_loaded.array).ravel()
            X, y = align_X_y(X_loaded.array, y)
            return X, y, X_loaded.feature_names

        raise ValueError(
            f"Unsupported y_path format: {cfg.y_path}. "
            "Supported: .mat, .npy, .npz, .csv/.tsv/.txt, .xlsx, .h5/.hdf5"
        )

    # No explicit y_path. For some formats, allow y in the same file.
    if x_lower.endswith((".h5", ".hdf5")) and cfg.y_key:
        try:
            y_loaded = read_array_auto(x_path, dataset_key=cfg.y_key)
            y = np.asarray(y_loaded.array).ravel()
            X, y = align_X_y(X_loaded.array, y)
            return X, y, X_loaded.feature_names
        except Exception:
            y = None

    if x_lower.endswith(".npz") and cfg.y_key:
        try:
            y_loaded = read_array_auto(x_path, npz_key=cfg.y_key)
            y = np.asarray(y_loaded.array).ravel()
            X, y = align_X_y(X_loaded.array, y)
            return X, y, X_loaded.feature_names
        except Exception:
            y = None

    X = coerce_X_only(X_loaded.array)
    return X, None, X_loaded.feature_names
