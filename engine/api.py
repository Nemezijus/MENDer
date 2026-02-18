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
    run_label_shuffle_baseline_from_cfg,
    save_model_bytes_from_cache,
    train_ensemble,
    train_supervised,
    train_unsupervised,
    tune_learning_curve,
    tune_validation_curve,
)

# Non-use-case helpers that are still part of the stable public surface.
from engine.io.readers import load_from_data_model
from engine.core.progress import ProgressCallback
from engine.io.artifacts.meta_models import ModelArtifactMetaDict
from engine.io.export.csv_export import ExportResult

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
    "run_label_shuffle_baseline_from_cfg",
    "load_from_data_model",
    "ProgressCallback",
    "ModelArtifactMetaDict",
    "ExportResult",
]
