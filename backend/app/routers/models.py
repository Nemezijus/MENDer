from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional

from ..models.v1.models import SaveModelRequest, LoadModelResponse
from ..services.model_persistence_service import save_model_service, load_model_service

router = APIRouter()


@router.post("/save", response_class=StreamingResponse, summary="Save the last trained model artifact")
async def save_model(req: SaveModelRequest):
    """
    Returns a binary joblib payload (Content-Disposition: attachment).
    The client is expected to download it (e.g., *.mend).
    """
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


@router.post("/load", response_model=LoadModelResponse, summary="Load a model artifact from an uploaded file")
async def load_model(file: UploadFile = File(...)):
    """
    Accepts a single uploaded artifact file and returns validated artifact meta.
    Also stores the fitted pipeline into a short-lived cache.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    data = await file.read()
    meta = load_model_service(data)

    return {"artifact": meta}
