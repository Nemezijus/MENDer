from fastapi import APIRouter

from engine.contracts.ensemble_run_config import EnsembleRunConfig

from ..models.v1.ensemble_models import EnsembleTrainRequest, EnsembleTrainResponse
from ..services.ensemble_train_service import train_ensemble

# NOTE: Versioning (/api/v1) is applied in backend/app/main.py.
router = APIRouter(prefix="/ensembles")


@router.post("/train", response_model=EnsembleTrainResponse)
def train_ensemble_endpoint(req: EnsembleTrainRequest) -> EnsembleTrainResponse:
    cfg = EnsembleRunConfig(
        data=req.data,
        split=req.split,
        scale=req.scale,
        features=req.features,
        ensemble=req.ensemble,
        eval=req.eval,
    )

    result = train_ensemble(cfg)
    return EnsembleTrainResponse(**result)
