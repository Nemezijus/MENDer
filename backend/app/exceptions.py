"""Backend-only exception types.

These are used to keep service code HTTP-agnostic while still allowing routers
or global exception handlers to map errors to appropriate HTTP responses.
"""

from __future__ import annotations


class ModelArtifactCacheGoneError(Exception):
    """Raised when the runtime model-artifact cache is missing/expired."""


class ModelArtifactValidationError(Exception):
    """Raised when a model artifact fails validation/deserialization."""


class ModelArtifactOperationError(Exception):
    """Raised for unexpected persistence operation failures (save/load)."""
