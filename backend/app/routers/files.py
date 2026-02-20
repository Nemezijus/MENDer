import os
from typing import Any, List

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.utils.upload_hashing import hash_and_stream_to_temp, safe_unlink
from backend.utils.upload_index import (
    update_index,
    lookup_display_name,
    get_index_filename,
)

from ..adapters.io.environment import get_upload_dir
from ..models.v1.files_models import UploadedFileInfo, FilesConstraints

# In Docker we set UPLOAD_DIR=/data/uploads; in dev we default to ./uploads
UPLOAD_DIR = get_upload_dir()
ALLOWED_EXTS = {".mat", ".npz", ".npy", ".csv", ".tsv", ".txt", ".h5", ".hdf5", ".xlsx"}

# NOTE: Versioning (/api/v1) is applied in backend/app/main.py.
router = APIRouter(prefix="/files")


@router.get("/ping")
def files_ping() -> dict[str, Any]:
    return {"ok": True, "dir": UPLOAD_DIR}


@router.get("/constraints", response_model=FilesConstraints)
def get_files_constraints() -> FilesConstraints:
    """Expose backend-owned IO constraints.

    Rationale:
      - Upload allowlist is a backend boundary concern.
      - Default X/y keys are applied in backend IO adapters.

    This helps keep the frontend from hardcoding boundary rules.
    """
    return FilesConstraints(
        upload_dir=UPLOAD_DIR,
        allowed_exts=sorted(ALLOWED_EXTS),
        data_default_keys={"x_key": "X", "y_key": "y"},
    )


@router.post("/upload", response_model=UploadedFileInfo)
async def upload_file(file: UploadFile = File(...)) -> UploadedFileInfo:
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


@router.get("/list", response_model=List[UploadedFileInfo])
def list_files() -> List[UploadedFileInfo]:
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
