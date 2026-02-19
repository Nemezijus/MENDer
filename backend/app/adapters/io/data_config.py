"""Construction of Engine DataModel configs from user inputs."""

from __future__ import annotations

from typing import Optional

from engine.contracts.run_config import DataModel

from .path_resolver import resolve_user_path


def build_data_config(
    npz_path: Optional[str],
    x_key: Optional[str],
    y_key: Optional[str],
    x_path: Optional[str],
    y_path: Optional[str],
) -> DataModel:
    npz_p = resolve_user_path(npz_path)
    x_p = resolve_user_path(x_path)
    y_p = resolve_user_path(y_path)
    return DataModel(
        npz_path=npz_p,
        x_path=x_p,
        y_path=y_p,
        x_key=x_key or "X",
        y_key=y_key or "y",
    )
