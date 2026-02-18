"""Public Engine API.

This module is the **stable public surface** for invoking business-logic use-cases.

Prefer importing from here instead of reaching into internal subpackages:

    from engine.api import train_supervised, predict

The backend and scripts may depend on this module.

The underlying implementations live under :mod:`engine.use_cases`.
"""

from __future__ import annotations

from engine.use_cases.facade import (
    export_decoder_outputs_to_csv,
    export_predictions_to_csv,
    grid_search,
    load_model_bytes_to_cache,
    predict,
    preview_pipeline,
    random_search,
    save_model_bytes_from_cache,
    train_ensemble,
    train_supervised,
    train_unsupervised,
    tune_learning_curve,
    tune_validation_curve,
)

__all__ = [
    "train_supervised",
    "train_unsupervised",
    "train_ensemble",
    "predict",
    "tune_learning_curve",
    "tune_validation_curve",
    "grid_search",
    "random_search",
    "preview_pipeline",
    "export_predictions_to_csv",
    "export_decoder_outputs_to_csv",
    "save_model_bytes_from_cache",
    "load_model_bytes_to_cache",
]
