from __future__ import annotations

from shared_schemas.run_config import RunConfig
from shared_schemas.tuning_configs import LearningCurveConfig
from utils.factories.tuning_factory import make_learning_curve_runner

from ..models.v1.tuning_models import (
    LearningCurveRequest,
    LearningCurveResponse,
)

from .common.result_coercion import to_payload


def compute_learning_curve(req: LearningCurveRequest) -> LearningCurveResponse:
    """
    Thin service wrapper for computing a learning curve.

    All heavy lifting (data loading, pipeline construction, sklearn.learning_curve)
    is handled in the business-logic layer (LearningCurveRunner); this service only
    maps the API models to shared configs and back.
    """

    # 1) Build the shared RunConfig used across training / tuning
    run_cfg = RunConfig(
        data=req.data,
        split=req.split,
        scale=req.scale,
        features=req.features,
        model=req.model,
        eval=req.eval,
    )

    # 2) Build the tuning-specific config
    lc_cfg = LearningCurveConfig(
        train_sizes=req.train_sizes,
        n_steps=req.n_steps,
        n_jobs=req.n_jobs,
    )

    # 3) Delegate to the tuning strategy
    runner = make_learning_curve_runner(run_cfg, lc_cfg)
    result = runner.run()

    # 4) Adapt to API response model
    return LearningCurveResponse(**to_payload(result))
