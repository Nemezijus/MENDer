# backend/app/adapters/io_adapter.py
from typing import Optional, Tuple
import numpy as np

from utils.configs.configs import DataConfig            # your config
from utils.strategies.interfaces import DataLoader      # typing only (optional)
from utils.strategies.data_loaders import AutoLoader    # not strictly needed, but ok to import
from utils.factories.data_loading_factory import make_data_loader  # your factory

class LoadError(Exception):
    pass

def build_data_config(
    npz_path: Optional[str],
    x_key: Optional[str],
    y_key: Optional[str],
    x_path: Optional[str],
    y_path: Optional[str],
) -> DataConfig:
    # Map exactly to your DataConfig fields
    return DataConfig(
        npz_path=npz_path,
        x_path=x_path,
        y_path=y_path,
        x_key=x_key or "X",
        y_key=y_key or "y",
    )

def load_X_y(
    npz_path: Optional[str],
    x_key: Optional[str],
    y_key: Optional[str],
    x_path: Optional[str],
    y_path: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        cfg = build_data_config(npz_path, x_key, y_key, x_path, y_path)
        loader = make_data_loader(cfg)   # AutoLoader under the hood
        X, y = loader.load()             # Uses NPZLoader/MatPairLoader + _coerce_shapes
        return X, y
    except Exception as e:
        # Normalize to our adapter exception for routers to map to HTTP errors cleanly
        raise LoadError(str(e))
