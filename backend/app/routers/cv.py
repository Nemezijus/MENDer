from fastapi import APIRouter
from ..models.v1.cv_models import CVRequest, CVResponse
from ..services.cv_service import run_kfold_cv
from utils.configs.configs import RunConfig

router = APIRouter()

@router.post("/crossval", response_model=CVResponse)
def crossval_endpoint(req: CVRequest):
    cfg = RunConfig(
        data=req.data,
        split=req.split,
        scale=req.scale,
        features=req.features,
        model=req.model,
        eval=req.eval,
    )
    result = run_kfold_cv(cfg)
    return CVResponse(**result)
