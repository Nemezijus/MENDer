"""Prediction / apply endpoints.

These routes are kept thin: they load arrays via the backend IO layer and
delegate execution to backend services (which call into ``engine.api``).
"""

from __future__ import annotations

from typing import Any, Union

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..models.v1.artifact_api_models import (
    ApplyModelRequest,
    ApplyModelResponse,
    ApplyModelExportRequest,
    ApplyUnsupervisedModelRequest,
    ApplyUnsupervisedModelResponse,
    ApplyUnsupervisedModelExportRequest,
    ExportDecoderOutputsRequest,
)
from ..models.v1.model_artifact import ModelArtifactMeta
from ..services.prediction_service import (
    apply_model_to_arrays,
    export_predictions_to_csv,
    export_decoder_outputs_to_csv,
)
from ..adapters.io.loader import load_X_optional_y

router = APIRouter()

# Sub-routers for clearer route ownership.
models_router = APIRouter(prefix="/models")
decoder_router = APIRouter(prefix="/decoder")

# Keep backward-compatible paths while improving internal structure.
router.include_router(models_router)
router.include_router(decoder_router)


def _load_prediction_data(data: Any) -> tuple[Any, Any]:
    """
    Load X and (optional) y for prediction/export endpoints.
    Uses the prediction-friendly IO helper where y is optional.
    """
    X, y = load_X_optional_y(
        data.npz_path,
        data.x_key,
        data.y_key,
        data.x_path,
        data.y_path,
    )
    return X, y


def _run_apply(
    req: Union[ApplyModelRequest, ApplyUnsupervisedModelRequest],
    *,
    export: bool = False,
) -> dict[str, Any]:
    """
    Shared execution for apply/export to keep router endpoints thin and consistent.
    """
    X, y = _load_prediction_data(req.data)

    # Allow apply/export to override evaluation/decoder settings without
    # mutating the stored artifact meta.
    artifact_meta = req.artifact_meta
    # Supervised apply/export allows an EvalModel override.
    if getattr(req, "eval", None) is not None and isinstance(req, ApplyModelRequest):
        meta_dict = req.artifact_meta.model_dump()
        meta_dict["eval"] = req.eval.model_dump()
        artifact_meta = ModelArtifactMeta(**meta_dict)

    if export:
        return export_predictions_to_csv(
            artifact_uid=req.artifact_uid,
            artifact_meta=artifact_meta,
            X=X,
            y=y,
            filename=getattr(req, "filename", None),
        )
    return apply_model_to_arrays(
        artifact_uid=req.artifact_uid,
        artifact_meta=artifact_meta,
        X=X,
        y=y,
    )


@models_router.post("/apply", response_model=Union[ApplyModelResponse, ApplyUnsupervisedModelResponse])
def apply_model_endpoint(
    req: Union[ApplyModelRequest, ApplyUnsupervisedModelRequest],
) -> Union[ApplyModelResponse, ApplyUnsupervisedModelResponse]:
    """
    Apply an existing model artifact to a new dataset.
    """
    result = _run_apply(req, export=False)
    if isinstance(result, dict) and result.get("task") == "unsupervised":
        return ApplyUnsupervisedModelResponse(**result)
    return ApplyModelResponse(**result)


@models_router.post("/apply/export")
def export_predictions_endpoint(
    req: Union[ApplyModelExportRequest, ApplyUnsupervisedModelExportRequest],
) -> StreamingResponse:
    """
    Export predictions as CSV for a given artifact + dataset.
    """
    export_result = _run_apply(req, export=True)

    headers = {
        "Content-Disposition": f'attachment; filename="{export_result["filename"]}"',
        "X-MENDER-Size": str(export_result["size"]),
    }

    return StreamingResponse(
        content=iter([export_result["content"]]),
        media_type=export_result["mime_type"],
        headers=headers,
    )


@decoder_router.post("/export")
def export_decoder_outputs_endpoint(req: ExportDecoderOutputsRequest) -> StreamingResponse:
    """Export cached evaluation (decoder) outputs as CSV.

    The `artifact_uid` must be obtained from a training endpoint response.
    """
    export_result = export_decoder_outputs_to_csv(
        artifact_uid=req.artifact_uid,
        filename=req.filename,
    )

    headers = {
        "Content-Disposition": f'attachment; filename="{export_result["filename"]}"',
        "X-MENDER-Size": str(export_result["size"]),
    }

    return StreamingResponse(
        content=iter([export_result["content"]]),
        media_type=export_result["mime_type"],
        headers=headers,
    )
