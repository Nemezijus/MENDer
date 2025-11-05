import os, uuid, re
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

# In Docker we set UPLOAD_DIR=/data/uploads; in dev we default to ./uploads
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.abspath("./uploads"))
ALLOWED_EXTS = {".mat", ".npz", ".npy", ".csv", ".txt"}

router = APIRouter()

class UploadedFileInfo(BaseModel):
    path: str           # filesystem path you can pass back into your loaders
    original_name: str
    saved_name: str

_slug = re.compile(r"[^a-zA-Z0-9._-]+")
def _safe_name(name: str) -> str:
    name = name.strip().replace(" ", "_")
    return _slug.sub("-", name)

@router.get("/files/ping")
def files_ping():
    return {"ok": True, "dir": UPLOAD_DIR}

@router.post("/files/upload", response_model=UploadedFileInfo)
async def upload_file(file: UploadFile = File(...)):
    orig = (file.filename or "upload").strip()
    stem, ext = os.path.splitext(orig)
    ext = ext.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported extension: {ext or '(none)'}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    uid = uuid.uuid4().hex[:12]
    saved_name = f"{uid}-{_safe_name(stem)}{ext}"
    dest = os.path.join(UPLOAD_DIR, saved_name)

    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(1_048_576)
                if not chunk:
                    break
                out.write(chunk)
    finally:
        await file.close()

    return UploadedFileInfo(path=dest, original_name=orig, saved_name=saved_name)

@router.get("/files/list", response_model=List[UploadedFileInfo])
def list_files():
    if not os.path.isdir(UPLOAD_DIR):
        return []
    items: List[UploadedFileInfo] = []
    for fn in sorted(os.listdir(UPLOAD_DIR)):
        p = os.path.join(UPLOAD_DIR, fn)
        if os.path.isfile(p):
            items.append(UploadedFileInfo(path=p, original_name=fn))
    return items
