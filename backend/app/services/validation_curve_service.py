"""Backend tuning service: validation curve.

Segment 12B: delegate orchestration to the Engine API (engine.api).
"""

from __future__ import annotations

from engine.api import tune_validation_curve as bl_tune_validation_curve

from engine.contracts.run_config import RunConfig
from engine.contracts.tuning_configs import ValidationCurveConfig

from ..models.v1.tuning_models import ValidationCurveRequest, ValidationCurveResponse

from .common.result_coercion import to_payload


def compute_validation_curve(req: ValidationCurveRequest) -> ValidationCurveResponse:
    """Compute a validation curve for the given request."""

    run_cfg = RunConfig(
        data=req.data,
        split=req.split,
        scale=req.scale,
        features=req.features,
        model=req.model,
        eval=req.eval,
    )

    vc_cfg = ValidationCurveConfig(
        param_name=req.param_name,
        param_range=req.param_range,
        n_jobs=req.n_jobs,
    )

    result = bl_tune_validation_curve(run_cfg, vc_cfg)
    return ValidationCurveResponse(**to_payload(result))
