"""Backend tuning service: learning curve.

Segment 12B: delegate orchestration to the Engine façade.
"""

from __future__ import annotations

from engine.use_cases.facade import tune_learning_curve as bl_tune_learning_curve

from engine.contracts.run_config import RunConfig
from engine.contracts.tuning_configs import LearningCurveConfig

from ..models.v1.tuning_models import LearningCurveRequest, LearningCurveResponse

from .common.result_coercion import to_payload


def compute_learning_curve(req: LearningCurveRequest) -> LearningCurveResponse:
    """Compute a learning curve for the given request."""

    run_cfg = RunConfig(
        data=req.data,
        split=req.split,
        scale=req.scale,
        features=req.features,
        model=req.model,
        eval=req.eval,
    )

    lc_cfg = LearningCurveConfig(
        train_sizes=req.train_sizes,
        n_steps=req.n_steps,
        n_jobs=req.n_jobs,
    )

    result = bl_tune_learning_curve(run_cfg, lc_cfg)
    return LearningCurveResponse(**to_payload(result))
