# backend/app/services/prediction_service.py
from __future__ import annotations

"""Prediction service (backend boundary).

Backend responsibilities:
- load X/y from disk (via io_adapter)
- call Engine API use-cases
- shape response models

Engine responsibilities:
- pipeline loading (cache/store)
- prediction/decoder orchestration
- export table assembly
"""

from typing import Any, Optional

from engine.api import (
    export_decoder_outputs_to_csv as bl_export_decoder_outputs_to_csv,
    export_predictions_to_csv as bl_export_predictions_to_csv,
    predict as bl_predict,
)

from engine.contracts.eval_configs import EvalModel
from engine.io.artifacts.meta_models import ModelArtifactMetaDict
from engine.io.export.csv_export import ExportResult

from .common.result_coercion import to_payload


def apply_model_to_arrays(
    *,
    artifact_uid: str,
    artifact_meta: ModelArtifactMetaDict,
    X: Any,
    y: Optional[Any] = None,
    eval_override: Optional[EvalModel] = None,
    max_preview_rows: int = 100,
) -> dict:
    """Apply a model (from cache/store) to arrays and return a JSON-friendly payload."""

    result = bl_predict(
        artifact_uid=artifact_uid,
        artifact_meta=artifact_meta,
        X=X,
        y=y,
        eval_override=eval_override,
        max_preview_rows=max_preview_rows,
    )

    return to_payload(result)


def export_predictions_to_csv(
    *,
    artifact_uid: str,
    artifact_meta: ModelArtifactMetaDict,
    X: Any,
    y: Optional[Any] = None,
    filename: Optional[str] = None,
    eval_override: Optional[EvalModel] = None,
) -> ExportResult:
    """Export predictions for the given artifact and dataset."""

    return bl_export_predictions_to_csv(
        artifact_uid=artifact_uid,
        artifact_meta=artifact_meta,
        X=X,
        y=y,
        filename=filename,
        eval_override=eval_override,
    )


def export_decoder_outputs_to_csv(*, artifact_uid: str, filename: Optional[str] = None) -> ExportResult:
    """Export cached evaluation (decoder) outputs for a previously trained artifact."""

    return bl_export_decoder_outputs_to_csv(artifact_uid=artifact_uid, filename=filename)
