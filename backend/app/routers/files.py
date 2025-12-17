import os
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.utils.upload_hashing import hash_and_stream_to_temp, safe_unlink
from backend.utils.upload_index import (
    update_index,
    lookup_display_name,
    get_index_filename,
)

# In Docker we set UPLOAD_DIR=/data/uploads; in dev we default to ./uploads
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.abspath("./uploads"))
ALLOWED_EXTS = {".mat", ".npz", ".npy", ".csv", ".txt"}

router = APIRouter()


class UploadedFileInfo(BaseModel):
    path: str  # filesystem path you can pass back into your loaders
    original_name: str
    saved_name: str


@router.get("/files/ping")
def files_ping():
    return {"ok": True, "dir": UPLOAD_DIR}


@router.post("/files/upload", response_model=UploadedFileInfo)
async def upload_file(file: UploadFile = File(...)):
    """Upload a single file into UPLOAD_DIR using content-addressed storage.

    Disk storage rule:
        saved_name = "<sha256><ext>"

    Metadata rule (JSON index):
        Keep track of original filenames used for the same content, so the UI can
        display meaningful names even though the stored filename is hashed.
    """
    orig = (file.filename or "upload").strip()
    _, ext = os.path.splitext(orig)
    ext = ext.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported extension: {ext or '(none)'}")

    try:
        digest, tmp_path = await hash_and_stream_to_temp(UPLOAD_DIR, file)
    finally:
        await file.close()

    saved_name = f"{digest}{ext}"
    dest = os.path.join(UPLOAD_DIR, saved_name)

    # If identical content already exists, discard temp and reuse.
    if os.path.exists(dest):
        safe_unlink(tmp_path)
        try:
            size_bytes = os.path.getsize(dest)
        except OSError:
            size_bytes = None
        update_index(
            UPLOAD_DIR,
            digest=digest,
            ext=ext,
            saved_name=saved_name,
            original_name=orig,
            size_bytes=size_bytes,
        )
        return UploadedFileInfo(path=dest, original_name=orig, saved_name=saved_name)

    # Atomic move into place.
    os.replace(tmp_path, dest)

    try:
        size_bytes = os.path.getsize(dest)
    except OSError:
        size_bytes = None

    update_index(
        UPLOAD_DIR,
        digest=digest,
        ext=ext,
        saved_name=saved_name,
        original_name=orig,
        size_bytes=size_bytes,
    )

    return UploadedFileInfo(path=dest, original_name=orig, saved_name=saved_name)


@router.get("/files/list", response_model=List[UploadedFileInfo])
def list_files():
    if not os.path.isdir(UPLOAD_DIR):
        return []

    index_fn = get_index_filename()

    items: List[UploadedFileInfo] = []
    for fn in sorted(os.listdir(UPLOAD_DIR)):
        p = os.path.join(UPLOAD_DIR, fn)

        if not os.path.isfile(p):
            continue

        # Skip internal index + temp files
        if fn == index_fn or fn.endswith(".tmp") or fn.startswith(".tmp-"):
            continue

        # Only list allowed data file types
        stem, ext = os.path.splitext(fn)
        ext_l = ext.lower()
        if ext_l not in ALLOWED_EXTS:
            continue

        # stem is expected to be the digest for canonical files "<digest><ext>"
        digest = stem
        display = lookup_display_name(UPLOAD_DIR, digest=digest, ext=ext_l) or fn

        items.append(
            UploadedFileInfo(
                path=p,
                original_name=display,
                saved_name=fn,
            )
        )

    return items
