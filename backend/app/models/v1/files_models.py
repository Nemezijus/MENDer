"""API models for file upload/list endpoints."""

from __future__ import annotations

from typing import List, Dict

from pydantic import BaseModel


class UploadedFileInfo(BaseModel):
    """Metadata returned by the file upload/list endpoints."""

    path: str  # filesystem path you can pass back into your loaders
    original_name: str
    saved_name: str


class FilesConstraints(BaseModel):
    """Backend-owned constraints for the file/data boundary.

    This intentionally lives in the backend (not Engine) because it describes
    HTTP/boundary behavior: allowlisted upload extensions and backend defaults
    applied in the IO adapters.
    """

    upload_dir: str
    allowed_exts: List[str]
    data_default_keys: Dict[str, str]
