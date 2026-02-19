# backend/app/routers/tuning.py
from __future__ import annotations

from fastapi import APIRouter

from ..models.v1.tuning_models import (
    LearningCurveRequest,
    LearningCurveResponse,
    ValidationCurveRequest,
    ValidationCurveResponse,
    GridSearchRequest,
    GridSearchResponse,
    RandomSearchRequest,
    RandomSearchResponse,
)
from ..services.learning_curve_service import compute_learning_curve
from ..services.validation_curve_service import compute_validation_curve
from ..services.grid_search_service import compute_grid_search
from ..services.random_search_service import compute_random_search


# NOTE: Versioning (/api/v1) is applied in backend/app/main.py.
router = APIRouter(prefix="/tuning")


@router.post("/learning-curve", response_model=LearningCurveResponse)
def learning_curve_endpoint(req: LearningCurveRequest) -> LearningCurveResponse:
    return compute_learning_curve(req)


@router.post("/validation-curve", response_model=ValidationCurveResponse)
def validation_curve_endpoint(
    req: ValidationCurveRequest,
) -> ValidationCurveResponse:
    return compute_validation_curve(req)


@router.post("/grid-search", response_model=GridSearchResponse)
def grid_search_endpoint(req: GridSearchRequest) -> GridSearchResponse:
    return compute_grid_search(req)


@router.post("/random-search", response_model=RandomSearchResponse)
def random_search_endpoint(req: RandomSearchRequest) -> RandomSearchResponse:
    return compute_random_search(req)
