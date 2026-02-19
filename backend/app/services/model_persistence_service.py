# backend/app/services/model_persistence_service.py
from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple

from ..exceptions import (
    ModelArtifactCacheGoneError,
    ModelArtifactOperationError,
    ModelArtifactValidationError,
)

from engine.api import (
    load_model_bytes_to_cache as bl_load_model_bytes_to_cache,
    save_model_bytes_from_cache as bl_save_model_bytes_from_cache,
)


def save_model_service(artifact_uid: str, artifact_meta: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    """Serialize the last trained model (from runtime cache) into bytes."""

    try:
        res = bl_save_model_bytes_from_cache(artifact_uid=artifact_uid, artifact_meta=artifact_meta)
        payload_bytes = res.content_bytes
    except ValueError as e:
        # Cache miss / expired is a user-facing state
        raise ModelArtifactCacheGoneError(str(e)) from e
    except Exception as e:
        # Keep HTTP concerns out of services; handlers map this to a 500.
        raise ModelArtifactOperationError(f"Save failed: {e}") from e

    sha256 = hashlib.sha256(payload_bytes).hexdigest()
    info = {
        "size": len(payload_bytes),
        "sha256": sha256,
    }
    return payload_bytes, info


def load_model_service(file_bytes: bytes) -> Dict[str, Any]:
    """Load a model artifact from bytes and place the pipeline into the runtime cache."""

    try:
        meta = bl_load_model_bytes_to_cache(file_bytes=file_bytes)
    except ValueError as e:
        raise ModelArtifactValidationError(str(e)) from e
    except Exception as e:
        raise ModelArtifactOperationError(f"Load failed: {e}") from e

    return meta
