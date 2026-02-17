from __future__ import annotations

from importlib import import_module
from typing import Any


def import_xgboost() -> Any:
    """Import xgboost lazily.

    This is the single dependency boundary for xgboost. Keeping it isolated makes
    the rest of the codebase import-safe even when xgboost isn't installed.
    """

    try:
        return import_module("xgboost")
    except Exception as e:
        raise ImportError(
            "XGBoost is not installed. Install the optional dependency 'xgboost' to use XGBoost ensembles."
        ) from e
