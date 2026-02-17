from __future__ import annotations

import warnings
from typing import Any, Optional

from engine.io.artifacts.store import ArtifactStore
from engine.runtime.caches.artifact_cache import artifact_cache
from engine.use_cases.artifacts import load_model_from_store


def load_pipeline(*, artifact_uid: str, store: Optional[ArtifactStore]) -> Any:
    """Load a trained pipeline from runtime cache or from an ArtifactStore."""

    pipeline = artifact_cache.get(artifact_uid)
    if pipeline is not None:
        return pipeline

    if store is None:
        raise ValueError(
            f"No cached model pipeline found for artifact_uid={artifact_uid!r}. "
            "Train a model in this process, or provide an ArtifactStore to load it."
        )

    pipeline, _meta = load_model_from_store(store, artifact_uid)

    try:
        artifact_cache.put(artifact_uid, pipeline)
    except Exception as e:
        warnings.warn(
            f"artifact_cache.put failed for artifact_uid={artifact_uid!r}: {type(e).__name__}: {e}",
            RuntimeWarning,
        )

    return pipeline
