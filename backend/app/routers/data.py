# backend/app/routers/data.py
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
import os
import shutil

router = APIRouter()

UPLOAD_DIR = os.path.join(os.getcwd(), "backend", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----- INSPECT -----
class InspectRequest(BaseModel):
    x_path: str | None = None
    y_path: str | None = None
    npz_path: str | None = None
    x_key: str = "X"
    y_key: str = "y"

@router.post("/data/inspect")
def inspect_endpoint(req: InspectRequest):
    from ..services.data_service import inspect_data
    # data_service.inspect_data expects a single request-like object
    return inspect_data(req)

# ----- UPLOAD -----
UPLOAD_DIR = os.path.join(os.getcwd(), "backend", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/data/upload")
async def upload_data(
    x_file: UploadFile | None = File(None),
    y_file: UploadFile | None = File(None),
):
    """
    Accept 1 or 2 files. Scenarios:
    - Single .npz containing both X,y (keys 'X'/'y' assumed unless changed later)
    - Single .mat containing X,y (keys 'X'/'y' assumed unless changed later)
    - Two files (X.mat/npz and y.mat/npz)
    Returns only saved server paths + default keys. The frontend then calls /data/inspect.
    """
    def _save(upload: UploadFile) -> str:
        dst = os.path.join(UPLOAD_DIR, upload.filename)
        with open(dst, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        return dst

    # Two files: treat as X and y separately
    if x_file and y_file:
        saved_x_path = _save(x_file)
        saved_y_path = _save(y_file)
        return {
            "saved": {
                "x_path": saved_x_path,
                "y_path": saved_y_path,
                "npz_path": None,
                "x_key": "X",
                "y_key": "y",
            }
        }

    # Single file
    if x_file and not y_file:
        path = _save(x_file)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            return {
                "saved": {
                    "x_path": None,
                    "y_path": None,
                    "npz_path": path,
                    "x_key": "X",
                    "y_key": "y",
                }
            }
        # assume .mat with X/y inside
        return {
            "saved": {
                "x_path": path,
                "y_path": None,
                "npz_path": None,
                "x_key": "X",
                "y_key": "y",
            }
        }

    return {"detail": "No files received"}
