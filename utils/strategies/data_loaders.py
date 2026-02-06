from __future__ import annotations

"""DataLoader implementations.

These loaders are responsible for:
- selecting the correct file reader based on file extension
- loading X and optional y
- performing minimal shape/orientation coercion

For tabular formats (csv/tsv/txt/xlsx), we also preserve feature names (when present)
as ``loader.feature_names`` for downstream visualization.

Implementation note
-------------------
The canonical parsing adapters live under ``engine.io.readers`` and shape utilities under
``engine.core.shapes``. This module remains as a compatibility layer during the
`utils` -> `engine` migration.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from engine.contracts.run_config import DataModel
from engine.core.shapes import coerce_X_only, align_X_y
from engine.io.readers import (
    load_mat_variable,
    load_npy_array,
    load_npz_array,
    load_delimited_table,
    load_xlsx_table,
    load_hdf5_dataset,
)


# --------------------------- legacy aliases -----------------------------


def _coerce_X_only(X: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for :func:`engine.core.shapes.coerce_X_only`."""
    return coerce_X_only(X)


def _coerce_shapes(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for :func:`engine.core.shapes.align_X_y`."""
    return align_X_y(X, y)


# ------------------------------ loaders ---------------------------------


@dataclass
class NPZLoader:
    cfg: DataModel
    feature_names: Optional[list[str]] = None

    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        path = self.cfg.npz_path or self.cfg.x_path
        if not path:
            raise ValueError("NPZLoader needs npz_path (or x_path pointing to .npz).")
        if not self.cfg.x_key:
            raise ValueError("NPZLoader requires x_key.")

        X_loaded = load_npz_array(path, key=self.cfg.x_key)
        X = np.asarray(X_loaded.array)

        y: Optional[np.ndarray] = None
        if self.cfg.y_key:
            try:
                y_loaded = load_npz_array(path, key=self.cfg.y_key)
                y = np.asarray(y_loaded.array).ravel()
                return _coerce_shapes(X, y)
            except KeyError:
                y = None

        X = _coerce_X_only(X)
        return X, None


@dataclass
class MatPairLoader:
    cfg: DataModel
    feature_names: Optional[list[str]] = None

    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.cfg.x_path:
            raise ValueError("MatPairLoader needs x_path.")
        X = np.asarray(load_mat_variable(self.cfg.x_path))

        if not self.cfg.y_path:
            return _coerce_X_only(X), None

        y = np.asarray(load_mat_variable(self.cfg.y_path)).ravel()
        return _coerce_shapes(X, y)


@dataclass
class NPYPairLoader:
    cfg: DataModel
    feature_names: Optional[list[str]] = None

    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.cfg.x_path:
            raise ValueError("NPYPairLoader needs x_path.")
        X = load_npy_array(self.cfg.x_path).array

        if not self.cfg.y_path:
            return _coerce_X_only(X), None

        y = load_npy_array(self.cfg.y_path).array.ravel()
        return _coerce_shapes(X, y)


@dataclass
class DelimitedPairLoader:
    cfg: DataModel
    feature_names: Optional[list[str]] = None

    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.cfg.x_path:
            raise ValueError("DelimitedPairLoader needs x_path.")

        x_loaded = load_delimited_table(
            self.cfg.x_path,
            delimiter=self.cfg.delimiter,
            has_header=self.cfg.has_header,
            encoding=self.cfg.encoding,
        )
        self.feature_names = x_loaded.feature_names
        X = x_loaded.array

        if not self.cfg.y_path:
            return _coerce_X_only(X), None

        y_has_header = None if (self.cfg.has_header is True) else self.cfg.has_header
        y_loaded = load_delimited_table(
            self.cfg.y_path,
            delimiter=self.cfg.delimiter,
            has_header=y_has_header,
            encoding=self.cfg.encoding,
        )
        y = y_loaded.array.ravel()
        return _coerce_shapes(X, y)


@dataclass
class XlsxPairLoader:
    cfg: DataModel
    feature_names: Optional[list[str]] = None

    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.cfg.x_path:
            raise ValueError("XlsxPairLoader needs x_path.")

        x_loaded = load_xlsx_table(
            self.cfg.x_path,
            sheet_name=self.cfg.sheet_name,
            has_header=self.cfg.has_header,
        )
        self.feature_names = x_loaded.feature_names
        X = x_loaded.array

        if not self.cfg.y_path:
            return _coerce_X_only(X), None

        y_has_header = None if (self.cfg.has_header is True) else self.cfg.has_header
        y_loaded = load_xlsx_table(
            self.cfg.y_path,
            sheet_name=self.cfg.sheet_name,
            has_header=y_has_header,
        )
        y = y_loaded.array.ravel()
        return _coerce_shapes(X, y)


@dataclass
class H5Loader:
    cfg: DataModel
    feature_names: Optional[list[str]] = None

    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.cfg.x_path:
            raise ValueError("H5Loader needs x_path.")

        x_loaded = load_hdf5_dataset(self.cfg.x_path, dataset_key=self.cfg.x_key)
        X = x_loaded.array

        if self.cfg.y_path:
            y_loaded = load_hdf5_dataset(self.cfg.y_path, dataset_key=self.cfg.y_key)
            y = y_loaded.array.ravel()
            return _coerce_shapes(X, y)

        if self.cfg.y_key:
            try:
                y_loaded = load_hdf5_dataset(self.cfg.x_path, dataset_key=self.cfg.y_key)
                y = y_loaded.array.ravel()
                return _coerce_shapes(X, y)
            except Exception:
                pass

        return _coerce_X_only(X), None


@dataclass
class AutoLoader:
    cfg: DataModel
    feature_names: Optional[list[str]] = None

    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.cfg.npz_path:
            loader = NPZLoader(self.cfg)
            X, y = loader.load()
            self.feature_names = getattr(loader, "feature_names", None)
            return X, y

        if not self.cfg.x_path:
            raise ValueError("AutoLoader requires x_path (or npz_path).")

        x_lower = self.cfg.x_path.lower()

        if x_lower.endswith(".npz"):
            loader = NPZLoader(self.cfg)
        elif x_lower.endswith(".mat"):
            loader = MatPairLoader(self.cfg)
        elif x_lower.endswith(".npy"):
            loader = NPYPairLoader(self.cfg)
        elif x_lower.endswith((".csv", ".tsv", ".txt")):
            loader = DelimitedPairLoader(self.cfg)
        elif x_lower.endswith((".h5", ".hdf5")):
            loader = H5Loader(self.cfg)
        elif x_lower.endswith(".xlsx"):
            loader = XlsxPairLoader(self.cfg)
        else:
            raise ValueError(
                f"Unsupported data format for x_path: {self.cfg.x_path}. "
                "Supported: .mat, .npz, .npy, .csv/.tsv/.txt, .h5/.hdf5, .xlsx"
            )

        X, y = loader.load()
        self.feature_names = getattr(loader, "feature_names", None)
        return X, y
