from __future__ import annotations

"""Model artifact persistence endpoints.

Point 6 refactor notes:
  - Canonical endpoints are exposed under /models/*.
  - Legacy endpoints (/save, /load) are retained for backward compatibility.

Versioning (/api/v1) is applied in backend/app/main.py.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from ..models.v1.models import SaveModelRequest, LoadModelResponse
from ..services.model_persistence_service import save_model_service, load_model_service


# Canonical router
router = APIRouter(prefix="/models")

# Legacy router (do not add a prefix here; main.py will mount under /api/v1)
legacy_router = APIRouter()


def _build_save_response(req: SaveModelRequest) -> StreamingResponse:
    payload_bytes, info = save_model_service(
        artifact_uid=req.artifact_uid,
        artifact_meta=req.artifact_meta.model_dump(),  # convert Pydantic to plain dict
    )

    filename = req.filename or f"{req.artifact_meta.uid}.mend"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-MENDER-SHA256": info["sha256"],
        "X-MENDER-Size": str(info["size"]),
    }

    return StreamingResponse(
        content=iter([payload_bytes]),
        media_type="application/octet-stream",
        headers=headers,
    )


async def _load_model_file(file: UploadFile) -> LoadModelResponse:
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    data = await file.read()
    meta = load_model_service(data)

    return {"artifact": meta}


# ------------------------
# Canonical endpoints
# ------------------------


@router.post("/save", response_class=StreamingResponse, summary="Save the last trained model artifact")
async def save_model(req: SaveModelRequest):
    """Download a binary artifact payload (Content-Disposition: attachment).

    The client is expected to download it (e.g., *.mend).
    """

    return _build_save_response(req)


@router.post("/load", response_model=LoadModelResponse, summary="Load a model artifact from an uploaded file")
async def load_model(file: UploadFile = File(...)):
    """Accepts a single uploaded artifact file and returns validated artifact meta.

    Also stores the fitted pipeline into a short-lived cache.
    """

    return await _load_model_file(file)


# ------------------------
# Legacy endpoints
# ------------------------


@legacy_router.post(
    "/save",
    response_class=StreamingResponse,
    summary="(Legacy) Save the last trained model artifact",
    deprecated=True,
)
async def legacy_save_model(req: SaveModelRequest):
    return _build_save_response(req)


@legacy_router.post(
    "/load",
    response_model=LoadModelResponse,
    summary="(Legacy) Load a model artifact from an uploaded file",
    deprecated=True,
)
async def legacy_load_model(file: UploadFile = File(...)):
    return await _load_model_file(file)
