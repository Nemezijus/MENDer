from fastapi import APIRouter, HTTPException

from shared_schemas.ensemble_run_config import EnsembleRunConfig

from ..models.v1.ensemble_models import EnsembleTrainRequest, EnsembleTrainResponse
from ..services.ensemble_train_service import train_ensemble
from ..adapters.io_adapter import LoadError

router = APIRouter()


@router.post("/ensembles/train", response_model=EnsembleTrainResponse)
def train_ensemble_endpoint(req: EnsembleTrainRequest):
    cfg = EnsembleRunConfig(
        data=req.data,
        split=req.split,
        scale=req.scale,
        features=req.features,
        ensemble=req.ensemble,
        eval=req.eval,
    )

    try:
        result = train_ensemble(cfg)
        return EnsembleTrainResponse(**result)
    except LoadError as e:
        raise HTTPException(status_code=400, detail=f"Data load failed: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
