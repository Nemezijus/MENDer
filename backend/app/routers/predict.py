# backend/app/routers/predict.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..models.v1.models import (
    ApplyModelRequest,
    ApplyModelResponse,
    ApplyModelExportRequest,
    ExportDecoderOutputsRequest,
)
from ..models.v1.model_artifact import ModelArtifactMeta
from ..services.prediction_service import (
    apply_model_to_arrays,
    export_predictions_to_csv,
    export_decoder_outputs_to_csv,
)
from ..adapters.io_adapter import load_X_optional_y, LoadError

router = APIRouter()


def _load_prediction_data(data):
    """
    Load X and (optional) y for prediction/export endpoints.
    Uses the prediction-friendly IO helper where y is optional.
    """
    try:
        X, y = load_X_optional_y(
            data.npz_path,
            data.x_key,
            data.y_key,
            data.x_path,
            data.y_path,
        )
        return X, y
    except LoadError as e:
        raise HTTPException(status_code=400, detail=f"Data load failed: {e}") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _run_apply(req: ApplyModelRequest, *, export: bool = False):
    """
    Shared execution for apply/export to keep router endpoints thin and consistent.
    """
    X, y = _load_prediction_data(req.data)

    # Allow apply/export to override evaluation/decoder settings without
    # mutating the stored artifact meta.
    artifact_meta = req.artifact_meta
    if getattr(req, "eval", None) is not None:
        meta_dict = req.artifact_meta.model_dump()
        meta_dict["eval"] = req.eval.model_dump()
        artifact_meta = ModelArtifactMeta(**meta_dict)

    try:
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/models/apply", response_model=ApplyModelResponse)
def apply_model_endpoint(req: ApplyModelRequest):
    """
    Apply an existing model artifact to a new dataset.
    """
    result = _run_apply(req, export=False)
    return ApplyModelResponse(**result)


@router.post("/models/apply/export")
def export_predictions_endpoint(req: ApplyModelExportRequest):
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


@router.post("/decoder/export")
def export_decoder_outputs_endpoint(req: ExportDecoderOutputsRequest):
    """Export cached evaluation (decoder) outputs as CSV.

    The `artifact_uid` must be obtained from a training endpoint response.
    """
    try:
        export_result = export_decoder_outputs_to_csv(
            artifact_uid=req.artifact_uid,
            filename=req.filename,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    headers = {
        "Content-Disposition": f'attachment; filename="{export_result["filename"]}"',
        "X-MENDER-Size": str(export_result["size"]),
    }

    return StreamingResponse(
        content=iter([export_result["content"]]),
        media_type=export_result["mime_type"],
        headers=headers,
    )
