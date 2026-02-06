"""Compatibility shim.

The canonical implementation moved to :mod:`engine.runtime.caches.artifact_cache`.
"""

from engine.runtime.caches.artifact_cache import ArtifactCache, artifact_cache

__all__ = [
    "ArtifactCache",
    "artifact_cache",
]
