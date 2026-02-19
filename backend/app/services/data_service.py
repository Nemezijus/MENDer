# backend/app/services/data_service.py
"""Backend data inspection service.

This module is a thin boundary adapter:
- it loads X (and optional y) using backend IO adapters (path validation, docker/dev mapping)
- it delegates *all* business logic to the Engine via ``engine.api``
"""

from __future__ import annotations

from typing import Any, Dict

from engine.api import inspect_dataset

from ..adapters.io_adapter import load_X_optional_y


def _load_arrays(payload) -> tuple[Any, Any]:
    """Load X and optional y using backend-owned IO normalization."""

    X, y = load_X_optional_y(
        payload.npz_path,
        payload.x_key,
        payload.y_key,
        payload.x_path,
        payload.y_path,
        expected_n_features=getattr(payload, "expected_n_features", None),
    )
    return X, y


def inspect_data(payload) -> Dict[str, Any]:
    """TRAINING inspect (smart): X required, y optional.

    If y is missing, the engine will mark ``task_inferred="unsupervised"`` so the
    frontend can route the user into unsupervised learning.
    """

    X, y = _load_arrays(payload)
    return inspect_dataset(X=X, y=y, treat_missing_y_as_unsupervised=True)


def inspect_data_optional_y(payload) -> Dict[str, Any]:
    """PRODUCTION inspect: X required, y optional.

    If y is missing, the engine will leave ``task_inferred`` as ``None``.
    """

    X, y = _load_arrays(payload)
    return inspect_dataset(X=X, y=y, treat_missing_y_as_unsupervised=False)
