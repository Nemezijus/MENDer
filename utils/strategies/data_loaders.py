# utils/strategies/data_loaders.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from utils.configs.configs import DataConfig
from utils.parse.data_read import load_mat_variable  # you already use this

def _coerce_shapes(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] < X.shape[1] and X.shape[1] == y.shape[0]:
        X = X.T
    if X.shape[0] != y.shape[0]:
        n = min(X.shape[0], y.shape[0])
        X, y = X[:n], y[:n]
    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D; got {y.shape}")
    return X, y

@dataclass
class NPZLoader:
    cfg: DataConfig
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
    cfg: DataConfig
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        if not (self.cfg.x_path and self.cfg.y_path):
            raise ValueError("MatPairLoader needs both x_path and y_path.")
        X = np.asarray(load_mat_variable(self.cfg.x_path))
        y = np.asarray(load_mat_variable(self.cfg.y_path)).ravel()
        return _coerce_shapes(X, y)

@dataclass
class AutoLoader:
    cfg: DataConfig
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        # Prefer explicit npz if given
        if self.cfg.npz_path:
            return NPZLoader(self.cfg).load()
        # If x_path is .npz and y_path is None, treat as NPZ bundle
        if self.cfg.x_path and self.cfg.x_path.lower().endswith(".npz") and not self.cfg.y_path:
            return NPZLoader(self.cfg).load()
        # Fall back to MAT pair
        return MatPairLoader(self.cfg).load()
