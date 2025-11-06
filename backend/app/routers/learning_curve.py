# backend/app/routers/learning_curve.py
from fastapi import APIRouter
from ..models.v1.learning_curve_models import LearningCurveRequest, LearningCurveResponse
from ..services.learning_curve_service import compute_learning_curve

router = APIRouter()

@router.post("/learning-curve", response_model=LearningCurveResponse)
def learning_curve_endpoint(req: LearningCurveRequest):
    return compute_learning_curve(req)
