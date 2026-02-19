"""FastAPI startup registration.

Keep import-time side effects out of routers/modules. Any filesystem setup,
warmups, or other initialization should be registered here.
"""

from __future__ import annotations

import os

from fastapi import FastAPI

from .adapters.io.environment import get_upload_dir


def register_startup(app: FastAPI) -> None:
    """Register startup hooks on the provided FastAPI app."""

    @app.on_event("startup")
    async def _ensure_runtime_directories() -> None:
        # In Docker we typically set UPLOAD_DIR=/data/uploads; in dev default to ./uploads
        os.makedirs(get_upload_dir(), exist_ok=True)
