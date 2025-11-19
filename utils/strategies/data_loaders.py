from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from shared_schemas.run_config import DataModel
from utils.parse.data_read import load_mat_variable

def _coerce_shapes(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    # Basic sanity
    if X.ndim == 1:
        X = X[:, None]
    if y.ndim != 1:
        raise ValueError(f"y must be 1D; got {y.shape}")

    n_rows, n_cols = X.shape
    n_y = y.shape[0]

    # Report what we got
    print(f"[INFO] Loaded X shape = ({n_rows}, {n_cols}); y length = {n_y}")

    # Decide orientation explicitly:
    # Case A: rows align with y -> rows are samples (preferred)
    if n_rows == n_y:
        print("[INFO] Using rows as observations (n_samples = rows, n_features = columns).")
        # no change

    # Case B: columns align with y -> columns are samples -> transpose
    elif n_cols == n_y:
        print("[INFO] Using COLUMNS as observations (X.T); transposing to (n_samples, n_features).")
        X = X.T
        n_rows, n_cols = X.shape  # update to post-transpose

    # Neither dimension matches y -> this is ambiguous; fail loudly
    else:
        raise ValueError(
            "Cannot align X with y: neither X.shape[0] nor X.shape[1] equals len(y). "
            f"Got X={X.shape}, len(y)={n_y}. Please check your inputs."
        )

    # Final assert
    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got {X.shape}")
    if X.shape[0] != y.shape[0]:
        # do NOT silently truncate — this hides bugs
        raise ValueError(
            f"After orientation, X and y length mismatch: X.shape[0]={X.shape[0]} vs len(y)={y.shape[0]}"
        )

    print(f"[INFO] Finalized dataset: {X.shape[0]} observations, {X.shape[1]} features.")
    return X, y


@dataclass
class NPZLoader:
    cfg: DataModel
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        path = self.cfg.npz_path or self.cfg.x_path
        if not path:
            raise ValueError("NPZLoader needs npz_path (or x_path pointing to .npz).")
        data = np.load(path, allow_pickle=True)
        if self.cfg.x_key not in data or self.cfg.y_key not in data:
            raise KeyError(
                f"Keys '{self.cfg.x_key}'/'{self.cfg.y_key}' not in {path}. "
                f"Available: {list(data.keys())}"
            )
        X = np.asarray(data[self.cfg.x_key])
        y = np.asarray(data[self.cfg.y_key]).ravel()
        return _coerce_shapes(X, y)

@dataclass
class MatPairLoader:
    cfg: DataModel
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        if not (self.cfg.x_path and self.cfg.y_path):
            raise ValueError("MatPairLoader needs both x_path and y_path.")
        X = np.asarray(load_mat_variable(self.cfg.x_path))
        y = np.asarray(load_mat_variable(self.cfg.y_path)).ravel()
        return _coerce_shapes(X, y)

@dataclass
class AutoLoader:
    cfg: DataModel
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        # Prefer explicit npz if given
        if self.cfg.npz_path:
            return NPZLoader(self.cfg).load()
        # If x_path is .npz and y_path is None, treat as NPZ bundle
        if self.cfg.x_path and self.cfg.x_path.lower().endswith(".npz") and not self.cfg.y_path:
            return NPZLoader(self.cfg).load()
        # Fall back to MAT pair
        return MatPairLoader(self.cfg).load()
