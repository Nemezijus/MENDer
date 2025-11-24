from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..models.v1.models import (
    ApplyModelRequest,
    ApplyModelResponse,
    ApplyModelExportRequest,
)
from ..services.prediction_service import (
    apply_model_to_arrays,
    export_predictions_to_csv,
)
from ..adapters.io_adapter import load_X_optional_y, LoadError

router = APIRouter()


@router.post("/models/apply", response_model=ApplyModelResponse)
def apply_model_endpoint(req: ApplyModelRequest):
    """
    Apply an existing model artifact to a new dataset.

    Flow:
      1. Load X (and optional y) using a prediction-friendly loader.
      2. Call prediction_service.apply_model_to_arrays with the cached pipeline.
      3. Wrap the result in ApplyModelResponse.
    """
    # 1) Load data via prediction-friendly IO helper (y is optional)
    try:
        data = req.data
        X, y = load_X_optional_y(
            data.npz_path,
            data.x_key,
            data.y_key,
            data.x_path,
            data.y_path,
        )
    except LoadError as e:
        raise HTTPException(status_code=400, detail=f"Data load failed: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 2) Delegate to prediction service (purely array-based)
    try:
        result = apply_model_to_arrays(
            artifact_uid=req.artifact_uid,
            artifact_meta=req.artifact_meta,
            X=X,
            y=y,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3) Return typed response
    return ApplyModelResponse(**result)


@router.post("/models/apply/export")
def export_predictions_endpoint(req: ApplyModelExportRequest):
    """
    Export predictions as CSV for a given artifact + dataset.

    The request body mirrors ApplyModelRequest but adds an optional filename.
    The response is a streaming CSV file (no JSON envelope).
    """
    # 1) Load data (X, optional y)
    try:
        data = req.data
        X, y = load_X_optional_y(
            data.npz_path,
            data.x_key,
            data.y_key,
            data.x_path,
            data.y_path,
        )
    except LoadError as e:
        raise HTTPException(status_code=400, detail=f"Data load failed: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 2) Run prediction + export
    try:
        export_result = export_predictions_to_csv(
            artifact_uid=req.artifact_uid,
            artifact_meta=req.artifact_meta,
            X=X,
            y=y,
            filename=req.filename,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    headers = {
        "Content-Disposition": f'attachment; filename="{export_result["filename"]}"',
        "X-MENDER-Size": str(export_result["size"]),
    }

    return StreamingResponse(
        content=iter([export_result["content"]]),
        media_type=export_result["mime_type"],
        headers=headers,
    )
