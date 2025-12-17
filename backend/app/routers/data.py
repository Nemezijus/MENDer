# backend/app/routers/data.py
from fastapi import APIRouter
from pydantic import BaseModel
import os

router = APIRouter()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.abspath("./uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ----- INSPECT -----
class InspectRequest(BaseModel):
    x_path: str | None = None
    y_path: str | None = None
    npz_path: str | None = None
    x_key: str = "X"
    y_key: str = "y"
    expected_n_features: int | None = None


@router.post("/data/inspect")
def inspect_endpoint(req: InspectRequest):
    """
    TRAINING inspect (strict): requires y for separate-file workflows.
    """
    from ..services.data_service import inspect_data

    return inspect_data(req)


@router.post("/data/inspect_production")
def inspect_production_endpoint(req: InspectRequest):
    """
    PRODUCTION inspect (y optional): allows X-only so users can prepare unseen data
    without labels.
    """
    from ..services.data_service import inspect_data_optional_y

    return inspect_data_optional_y(req)
