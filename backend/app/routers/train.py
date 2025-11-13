from fastapi import APIRouter, HTTPException

from ..models.v1.train_models import TrainRequest, TrainResponse
from ..services.train_service import train
from ..adapters.io_adapter import LoadError
from utils.configs.configs import RunConfig

router = APIRouter()


@router.post("/train", response_model=TrainResponse)
def train_endpoint(req: TrainRequest):
    cfg = RunConfig(
        data=req.data,
        split=req.split,
        scale=req.scale,
        features=req.features,
        model=req.model,
        eval=req.eval,
    )

    try:
        result = train(cfg)
        return TrainResponse(**result)
    except LoadError as e:
        raise HTTPException(status_code=400, detail=f"Data load failed: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
