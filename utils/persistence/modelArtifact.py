"""Compatibility shim.

The canonical implementation moved to :mod:`engine.io.artifacts.meta`.
"""

from engine.io.artifacts.meta import ArtifactBuilderInput, build_model_artifact_meta

__all__ = [
    "ArtifactBuilderInput",
    "build_model_artifact_meta",
]
