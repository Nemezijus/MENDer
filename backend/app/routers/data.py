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
