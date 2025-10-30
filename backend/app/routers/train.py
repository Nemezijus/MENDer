from fastapi import APIRouter, HTTPException
from ..models.v1.train_models import TrainRequest, TrainResponse
from ..services.train_service import train_once
from ..adapters.io_adapter import LoadError

router = APIRouter()

@router.post("/train", response_model=TrainResponse)
def train_endpoint(req: TrainRequest):
    """
    Run a single training session using your real MENDer factories:
    - Load & sanity-check data
    - Split (stratified or not)
    - Fit pipeline (scale → feat → clf)
    - Score and return confusion matrix
    """
    try:
        result = train_once(req)
        return TrainResponse(**result)
    except LoadError as e:
        raise HTTPException(status_code=400, detail=f"Data load failed: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
