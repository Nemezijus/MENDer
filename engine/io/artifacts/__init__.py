"""Artifact persistence for the Engine (Business Layer).

This package centralizes:
- how model artifacts are serialized/deserialized (joblib bytes)
- how artifact metadata is constructed (a dict compatible with backend ModelArtifactMeta)
- how artifacts are stored (ArtifactStore abstraction; filesystem default)

Business Layer rule of thumb:
- core computation must not depend on process-local caches.
- caches belong under :mod:`engine.runtime`.
"""

from .serialization import SCHEMA_VERSION, MAGIC_KEY, SaveResult, save_model_artifact, load_model_artifact
from .meta import ArtifactBuilderInput, build_model_artifact_meta
from .meta_models import ArtifactSummary, ModelArtifactMetaDict
from .store import ArtifactStore, StoredArtifact
from .filesystem_store import FileSystemArtifactStore, get_default_filesystem_store

__all__ = [
    "SCHEMA_VERSION",
    "MAGIC_KEY",
    "SaveResult",
    "save_model_artifact",
    "load_model_artifact",
    "ArtifactBuilderInput",
    "build_model_artifact_meta",
    "ArtifactSummary",
    "ModelArtifactMetaDict",
    "ArtifactStore",
    "StoredArtifact",
    "FileSystemArtifactStore",
    "get_default_filesystem_store",
]
