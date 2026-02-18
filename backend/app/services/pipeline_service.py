# backend/app/services/pipeline_service.py
from __future__ import annotations

"""Pipeline preview service.

The backend is responsible only for:
- translating request payloads into Engine contracts
- shaping the HTTP response

All orchestration (RNG + pipeline factory + introspection) lives in the Engine.
"""

from engine.api import preview_pipeline as bl_preview_pipeline

from engine.contracts.run_config import RunConfig, DataModel
from engine.contracts.scale_configs import ScaleModel
from engine.contracts.split_configs import SplitHoldoutModel


def preview_pipeline(payload):
    cfg = RunConfig(
        data=DataModel(),
        split=SplitHoldoutModel(mode="holdout"),
        scale=ScaleModel(**payload.scale.model_dump()),
        features=payload.features,
        model=payload.model,
        eval=payload.eval,
    )

    return bl_preview_pipeline(run_config=cfg)
