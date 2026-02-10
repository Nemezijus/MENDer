"""Backend ensemble training service.

Segment 12B: delegate orchestration to the Engine façade.

The backend remains responsible for HTTP concerns (error mapping) and API
request/response validation. All compute/orchestration lives in
``engine.use_cases``.
"""

from __future__ import annotations

from typing import Any, Dict

from engine.use_cases.facade import train_ensemble as bl_train_ensemble

from engine.contracts.ensemble_run_config import EnsembleRunConfig

from ..adapters.io_adapter import LoadError

from .common.result_coercion import to_payload


def _is_probable_load_error(exc: Exception) -> bool:
    """Best-effort classification of 'data load' failures."""

    if isinstance(exc, (FileNotFoundError, OSError, IOError)):
        return True

    msg = str(exc).lower()
    needles = (
        "npz",
        "npy",
        "mat",
        "h5",
        "hdf5",
        "csv",
        "file",
        "path",
        "no such file",
        "not found",
        "missing",
        "x_key",
        "y_key",
        "load",
        "dataset",
        "could not read",
        "cannot read",
    )
    return any(n in msg for n in needles)


def train_ensemble(cfg: EnsembleRunConfig) -> Dict[str, Any]:
    """Train an ensemble model.

    Delegates to :func:`engine.use_cases.facade.train_ensemble`.
    """

    try:
        result = bl_train_ensemble(cfg)
    except Exception as e:
        if _is_probable_load_error(e):
            raise LoadError(str(e)) from e
        raise

    return to_payload(result)
