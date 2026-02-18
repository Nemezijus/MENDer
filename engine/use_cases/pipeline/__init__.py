"""Pipeline-related use-cases.

The backend/UI can request a "dry" pipeline build (instantiate steps without fit)
for inspection/preview purposes.

This package keeps that orchestration inside the Engine so the backend does not
need to import factories or runtime helpers.
"""

from .preview import preview_pipeline

__all__ = ["preview_pipeline"]
