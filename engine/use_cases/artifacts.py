"""Use-cases for model artifact persistence.

These functions keep the Engine runnable without backend/frontend.
They depend on the :class:`~engine.io.artifacts.store.ArtifactStore` abstraction
and do not rely on any process-local caches.

Backend services may call these directly.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from engine.io.artifacts.serialization import SaveResult, load_model_artifact, save_model_artifact
from engine.io.artifacts.store import ArtifactStore, StoredArtifact


def save_model_bytes(pipeline: Any, meta: Dict[str, Any]) -> SaveResult:
    """Serialize a fitted pipeline + meta to bytes."""

    return save_model_artifact(pipeline, meta)


def load_model_bytes(file_bytes: bytes) -> Tuple[Any, Dict[str, Any]]:
    """Deserialize artifact bytes to (pipeline, meta)."""

    return load_model_artifact(file_bytes)


def save_model_to_store(
    store: ArtifactStore,
    pipeline: Any,
    meta: Dict[str, Any],
) -> Tuple[SaveResult, StoredArtifact]:
    """Serialize and persist to the provided ArtifactStore."""

    res = save_model_artifact(pipeline, meta)
    uid = meta.get("uid")
    if not uid:
        raise ValueError("Artifact meta missing 'uid'")

    ref = store.save(str(uid), res.content_bytes, meta=meta)
    return res, ref


def load_model_from_store(store: ArtifactStore, uid: str) -> Tuple[Any, Dict[str, Any]]:
    """Load bytes from a store and deserialize to (pipeline, meta)."""

    payload, _ = store.load(uid)
    pipeline, meta = load_model_artifact(payload)
    return pipeline, meta
