"""Backend ensemble training service.

Segment 12B: delegate orchestration to the Engine API (engine.api).

The backend remains responsible for HTTP concerns (error mapping) and API
request/response validation. All compute/orchestration lives in
``engine.use_cases``.
"""

from __future__ import annotations

from typing import Any, Dict

from engine.api import train_ensemble as bl_train_ensemble

from engine.contracts.ensemble_run_config import EnsembleRunConfig

from ..adapters.io_adapter import LoadError

from .common.error_classification import is_probable_load_error
from .common.result_coercion import to_payload


def train_ensemble(cfg: EnsembleRunConfig) -> Dict[str, Any]:
    """Train an ensemble model.

    Delegates to :func:`engine.api.train_ensemble`.
    """

    try:
        result = bl_train_ensemble(cfg)
    except Exception as e:
        if is_probable_load_error(e):
            raise LoadError(str(e)) from e
        raise

    return to_payload(result)
