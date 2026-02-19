"""Data-related endpoints.

These endpoints are intentionally thin: they validate the request payload, load
arrays using backend IO adapters, and delegate dataset inspection logic to the
Engine (BL) via ``engine.api``.
"""

from fastapi import APIRouter

from ..models.v1.data_models import DataInspectRequest, DataInspectResponse
from ..services.data_service import inspect_data, inspect_data_optional_y


# NOTE: Versioning (/api/v1) is applied in backend/app/main.py.
router = APIRouter(prefix="/data")


@router.post("/inspect", response_model=DataInspectResponse)
def inspect_endpoint(req: DataInspectRequest) -> DataInspectResponse:
    """TRAINING inspect (smart): X required, y optional.

    If y is missing, the response will set ``task_inferred="unsupervised"`` so the
    frontend can route the user into unsupervised learning without a separate
    upload flow.
    """

    return DataInspectResponse(**inspect_data(req))


@router.post("/inspect_production", response_model=DataInspectResponse)
def inspect_production_endpoint(req: DataInspectRequest) -> DataInspectResponse:
    """PRODUCTION inspect (y optional): allows X-only so users can prepare unseen data
    without labels.
    """

    return DataInspectResponse(**inspect_data_optional_y(req))
