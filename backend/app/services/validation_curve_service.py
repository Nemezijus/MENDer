# backend/app/services/validation_curve_service.py
from __future__ import annotations

from shared_schemas.run_config import RunConfig
from shared_schemas.tuning_configs import ValidationCurveConfig
from utils.factories.tuning_factory import make_validation_curve_runner

from ..models.v1.tuning_models import (
    ValidationCurveRequest,
    ValidationCurveResponse,
)

from .common.result_coercion import to_payload


def compute_validation_curve(
    req: ValidationCurveRequest,
) -> ValidationCurveResponse:
    """
    Thin service wrapper for computing a validation curve.

    All heavy lifting (data loading, pipeline construction,
    sklearn.validation_curve, etc.) is handled in the business-logic layer
    (ValidationCurveRunner). This service only maps the API models to shared
    configs and back.
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
    vc_cfg = ValidationCurveConfig(
        param_name=req.param_name,
        param_range=req.param_range,
        n_jobs=req.n_jobs,
    )

    # 3) Delegate to the tuning strategy
    runner = make_validation_curve_runner(run_cfg, vc_cfg)
    result = runner.run()

    # 4) Adapt to API response model
    return ValidationCurveResponse(**to_payload(result))
