from __future__ import annotations

"""DataModel convenience loader.

This module is responsible for interpreting :class:`engine.contracts.run_config.DataModel`
and loading X (and optional y) accordingly. It uses :func:`read_array_auto` for format parsing.

"""

from typing import Optional, Tuple

import numpy as np

from engine.contracts.run_config import DataModel
from engine.core.shapes import align_X_y, coerce_X_only

from ..base import LoadedArray
from .dispatch import read_array_auto


def load_from_data_model(cfg: DataModel) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[list[str]]]:
    """Load X and optional y using a :class:`~engine.contracts.run_config.DataModel`.

    Returns
    -------
    X : np.ndarray (2D)
    y : np.ndarray | None (1D)
    feature_names : list[str] | None
    """

    # Prefer explicit npz_path if provided
    if cfg.npz_path:
        return _load_from_single_npz(cfg)

    if not cfg.x_path:
        raise ValueError("DataModel requires x_path (or npz_path)")

    X_loaded = _load_X(cfg)
    y = _load_y(cfg, X_loaded)

    if y is None:
        X = coerce_X_only(X_loaded.array)
        return X, None, X_loaded.feature_names

    X, y_aligned = align_X_y(X_loaded.array, y)
    return X, y_aligned, X_loaded.feature_names


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_from_single_npz(cfg: DataModel) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[list[str]]]:
    npz_path = cfg.npz_path
    if not cfg.x_key:
        raise ValueError("NPZ loading requires x_key")

    X_loaded = read_array_auto(npz_path, npz_key=cfg.x_key)

    if cfg.y_key:
        try:
            y_loaded = read_array_auto(npz_path, npz_key=cfg.y_key)
            y = np.asarray(y_loaded.array).ravel()
            X, y = align_X_y(X_loaded.array, y)
            return X, y, X_loaded.feature_names
        except KeyError:
            # y_key missing in NPZ: treat as X-only
            pass

    X = coerce_X_only(X_loaded.array)
    return X, None, X_loaded.feature_names


def _load_X(cfg: DataModel) -> LoadedArray:
    x_path = cfg.x_path
    assert x_path is not None
    x_lower = x_path.lower()

    if x_lower.endswith(".npz"):
        if not cfg.x_key:
            raise ValueError("NPZ loading requires x_key")
        return read_array_auto(x_path, npz_key=cfg.x_key)

    if x_lower.endswith((".h5", ".hdf5")):
        return read_array_auto(x_path, dataset_key=cfg.x_key)

    if x_lower.endswith((".csv", ".tsv", ".txt")):
        return read_array_auto(
            x_path,
            delimiter=cfg.delimiter,
            has_header=cfg.has_header,
            encoding=cfg.encoding,
        )

    if x_lower.endswith(".xlsx"):
        return read_array_auto(
            x_path,
            sheet_name=cfg.sheet_name,
            has_header=cfg.has_header,
        )

    return read_array_auto(x_path)


def _load_y(cfg: DataModel, X_loaded: LoadedArray) -> Optional[np.ndarray]:
    # Explicit y_path takes precedence
    if cfg.y_path:
        return _load_y_from_path(cfg, X_loaded)

    # No explicit y_path. For some formats, allow y in the same file.
    x_path = cfg.x_path
    assert x_path is not None
    x_lower = x_path.lower()

    if x_lower.endswith((".h5", ".hdf5")) and cfg.y_key:
        try:
            y_loaded = read_array_auto(x_path, dataset_key=cfg.y_key)
            return np.asarray(y_loaded.array).ravel()
        except Exception as e:
            raise ValueError(
                f"Failed to load y_key={cfg.y_key!r} from HDF5 file {x_path!r}."
            ) from e

    if x_lower.endswith(".npz") and cfg.y_key:
        try:
            y_loaded = read_array_auto(x_path, npz_key=cfg.y_key)
            return np.asarray(y_loaded.array).ravel()
        except Exception as e:
            raise ValueError(f"Failed to load y_key={cfg.y_key!r} from NPZ file {x_path!r}.") from e

    return None


def _load_y_from_path(cfg: DataModel, X_loaded: LoadedArray) -> np.ndarray:
    y_path = cfg.y_path
    assert y_path is not None
    y_lower = y_path.lower()

    if y_lower.endswith(".npz"):
        if not cfg.y_key:
            raise ValueError("NPZ y_path requires y_key")
        y_loaded = read_array_auto(y_path, npz_key=cfg.y_key)
        return np.asarray(y_loaded.array).ravel()

    if y_lower.endswith((".h5", ".hdf5")):
        y_loaded = read_array_auto(y_path, dataset_key=cfg.y_key)
        return np.asarray(y_loaded.array).ravel()

    if y_lower.endswith((".csv", ".tsv", ".txt")):
        # y usually has no header; do not force cfg.has_header=True onto y.
        y_has_header = None if (cfg.has_header is True) else cfg.has_header
        y_loaded = read_array_auto(
            y_path,
            delimiter=cfg.delimiter,
            has_header=y_has_header,
            encoding=cfg.encoding,
        )
        return np.asarray(y_loaded.array).ravel()

    if y_lower.endswith(".xlsx"):
        y_has_header = None if (cfg.has_header is True) else cfg.has_header
        y_loaded = read_array_auto(
            y_path,
            sheet_name=cfg.sheet_name,
            has_header=y_has_header,
        )
        return np.asarray(y_loaded.array).ravel()

    if y_lower.endswith(".mat") or y_lower.endswith(".npy"):
        y_loaded = read_array_auto(y_path)
        return np.asarray(y_loaded.array).ravel()

    raise ValueError(
        f"Unsupported y_path format: {cfg.y_path}. "
        "Supported: .mat, .npy, .npz, .csv/.tsv/.txt, .xlsx, .h5/.hdf5"
    )
