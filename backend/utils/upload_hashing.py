"""Helpers for content-addressed uploads.

We stream uploaded files to a temporary file while computing a SHA-256 digest.
Callers can then atomically move the temp file into place using the digest as the filename.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from typing import Tuple

from fastapi import UploadFile


DEFAULT_CHUNK_SIZE = 1_048_576  # 1 MiB


async def hash_and_stream_to_temp(
    upload_dir: str,
    file: UploadFile,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    tmp_prefix: str = ".tmp-",
) -> Tuple[str, str]:
    """Stream an UploadFile to a temp file while computing SHA-256.

    Args:
        upload_dir: Destination directory for temp file (and later final file).
        file: FastAPI UploadFile to read from.
        chunk_size: Read size per iteration.
        tmp_prefix: Prefix for the temp filename.

    Returns:
        (hex_digest, temp_path)

    Notes:
        - The temp file is created inside upload_dir so that os.replace() is atomic.
        - This function does *not* close the UploadFile; callers should close it.
    """
    os.makedirs(upload_dir, exist_ok=True)

    tmp_name = f"{tmp_prefix}{uuid.uuid4().hex}"
    tmp_path = os.path.join(upload_dir, tmp_name)

    h = hashlib.sha256()
    with open(tmp_path, "wb") as out:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            out.write(chunk)

    return h.hexdigest(), tmp_path


def safe_unlink(path: str) -> None:
    """Best-effort file removal (no exception if it fails)."""
    try:
        os.remove(path)
    except OSError:
        pass
