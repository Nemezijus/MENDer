from fastapi import APIRouter
from typing import Union

from ..models.v1.train_models import (
    TrainRequest,
    TrainResponse,
    UnsupervisedTrainRequest,
    UnsupervisedTrainResponse,
)
from ..services.train_service import train, train_unsupervised

from engine.contracts.run_config import RunConfig
from engine.contracts.unsupervised_configs import UnsupervisedRunConfig

router = APIRouter()


@router.post("/train", response_model=Union[TrainResponse, UnsupervisedTrainResponse])
def train_endpoint(req: Union[TrainRequest, UnsupervisedTrainRequest]):
    """Train a model.

    Backward compatible:
      - Supervised requests use TrainRequest/TrainResponse
      - Unsupervised requests use UnsupervisedTrainRequest/UnsupervisedTrainResponse
    """

    is_unsupervised = getattr(req, "task", None) == "unsupervised"

    if is_unsupervised:
        cfg = UnsupervisedRunConfig(
            data=req.data,
            apply=req.apply,
            fit_scope=req.fit_scope,
            scale=req.scale,
            features=req.features,
            model=req.model,
            eval=req.eval,
            use_y_for_external_metrics=req.use_y_for_external_metrics,
            external_metrics=req.external_metrics,
        )
        result = train_unsupervised(cfg)
        return UnsupervisedTrainResponse(**result)

    cfg = RunConfig(
        data=req.data,
        split=req.split,
        scale=req.scale,
        features=req.features,
        model=req.model,
        eval=req.eval,
    )

    result = train(cfg)
    return TrainResponse(**result)
