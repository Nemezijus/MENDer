"""Cache-aware artifact persistence helpers.

These functions are intentionally *use-cases* (orchestrators) because they
coordinate:
- runtime cache access (process-local)
- artifact (de)serialization

Keeping this in the Engine enforces the BL/backend boundary:
backend must not import ``engine.runtime.*``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from engine.io.artifacts.serialization import SaveResult
from engine.use_cases.artifacts import load_model_bytes, save_model_bytes
from engine.runtime.caches.artifact_cache import artifact_cache


def save_model_bytes_from_cache(artifact_uid: str, artifact_meta: Dict[str, Any]) -> SaveResult:
    """Serialize a cached pipeline (by uid) + meta to bytes."""

    pipeline = artifact_cache.get(artifact_uid)
    if pipeline is None:
        raise ValueError(
            "Model is no longer available for saving (cache expired). "
            "Re-run training or load a saved model."
        )

    return save_model_bytes(pipeline, artifact_meta)


def load_model_bytes_to_cache(file_bytes: bytes) -> Dict[str, Any]:
    """Deserialize artifact bytes and put the pipeline into the runtime cache.

    Returns
    -------
    meta: dict
        The validated artifact meta extracted from the file.
    """

    pipeline, meta = load_model_bytes(file_bytes)

    uid = meta.get("uid")
    if not uid:
        raise ValueError("Artifact meta missing 'uid'")

    artifact_cache.put(str(uid), pipeline)
    return meta
