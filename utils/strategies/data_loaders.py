from __future__ import annotations

"""DataLoader implementations.

These loaders are responsible for:
- selecting the correct file reader based on file extension
- loading X and optional y
- performing minimal shape/orientation coercion

For tabular formats (csv/tsv/txt/xlsx), we also preserve feature names (when present)
as ``loader.feature_names`` for downstream visualization.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from shared_schemas.run_config import DataModel
from utils.parse.data_read import (
    load_mat_variable,
    load_npy_array,
    load_delimited_table,
    load_xlsx_table,
    load_hdf5_dataset,
)


def _coerce_X_only(X: np.ndarray) -> np.ndarray:
    """Basic coercion for X-only datasets (unsupervised)."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got {X.shape}")
    n_rows, n_cols = X.shape
    if n_rows < 1 or n_cols < 1:
        raise ValueError(f"X must have at least 1 sample and 1 feature; got {X.shape}")
    return X


def _coerce_shapes(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align X orientation to y.

    Preferred convention:
    - rows are samples, columns are features -> X.shape == (n_samples, n_features)

    If X is provided as (n_features, n_samples) and columns align with y, we transpose.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if X.ndim == 1:
        X = X[:, None]
    if y.ndim != 1:
        raise ValueError(f"y must be 1D; got {y.shape}")

    n_rows, n_cols = X.shape
    n_y = y.shape[0]

    if n_rows == n_y:
        pass
    elif n_cols == n_y:
        X = X.T
    else:
        raise ValueError(
            "Cannot align X with y: neither X.shape[0] nor X.shape[1] equals len(y). "
            f"Got X={X.shape}, len(y)={n_y}."
        )

    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"After orientation, X and y length mismatch: X.shape[0]={X.shape[0]} vs len(y)={y.shape[0]}"
        )

    return X, y


# ------------------------------ loaders ---------------------------------


@dataclass
class NPZLoader:
    cfg: DataModel
    feature_names: Optional[list[str]] = None

    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        path = self.cfg.npz_path or self.cfg.x_path
        if not path:
            raise ValueError("NPZLoader needs npz_path (or x_path pointing to .npz).")
        data = np.load(path, allow_pickle=True)
        if not self.cfg.x_key:
            raise ValueError("NPZLoader requires x_key.")
        if self.cfg.x_key not in data:
            raise KeyError(
                f"Key '{self.cfg.x_key}' not in {path}. Available: {list(data.keys())}"
            )
        X = np.asarray(data[self.cfg.x_key])

        y: Optional[np.ndarray] = None
        if self.cfg.y_key and self.cfg.y_key in data:
            y = np.asarray(data[self.cfg.y_key]).ravel()
            return _coerce_shapes(X, y)

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

        # y usually has no header; do not force cfg.has_header=True onto y.
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

        # y can be in separate file, or in the same file under y_key
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
                # If y_key is not present, treat as X-only.
                pass

        return _coerce_X_only(X), None


@dataclass
class AutoLoader:
    cfg: DataModel
    feature_names: Optional[list[str]] = None

    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Prefer explicit npz if given
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
