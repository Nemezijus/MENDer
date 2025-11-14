from __future__ import annotations

from typing import Tuple, Dict, Any
from fastapi import HTTPException, status

from utils.persistence.artifact_cache import artifact_cache
from utils.persistence.model_io import save_model_artifact, load_model_artifact


def save_model_service(artifact_uid: str, artifact_meta: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    """
    Fetch the fitted pipeline from the short-lived cache by artifact UID,
    serialize it together with the provided meta, and return the raw bytes
    and the (possibly augmented) meta dict.
    """
    pipeline = artifact_cache.get(artifact_uid)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Model is no longer available for saving (cache expired). Re-run training or load a saved model.",
        )

    result = save_model_artifact(pipeline, artifact_meta)  # returns SaveResult
    return result.content_bytes, {"size": result.size, "sha256": result.sha256}


def load_model_service(file_bytes: bytes) -> Dict[str, Any]:
    """
    Load a model artifact from bytes, validate it, place the pipeline into cache,
    and return the artifact meta for UI display.
    """
    pipeline, meta = load_model_artifact(file_bytes)

    uid = meta.get("uid")
    if not uid:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Artifact meta missing 'uid'")

    # Put loaded pipeline into cache so it can be used for 'apply' (future) or re-save.
    artifact_cache.put(uid, pipeline)

    return meta
