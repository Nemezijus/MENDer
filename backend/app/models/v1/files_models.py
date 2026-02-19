"""API models for file upload/list endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class UploadedFileInfo(BaseModel):
    """Metadata returned by the file upload/list endpoints."""

    path: str  # filesystem path you can pass back into your loaders
    original_name: str
    saved_name: str
