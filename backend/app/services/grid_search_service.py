"""Backend tuning service: grid search.

Segment 12B: delegate orchestration to the Engine API (engine.api).
"""

from __future__ import annotations

from engine.api import grid_search as bl_grid_search

from engine.contracts.run_config import RunConfig
from engine.contracts.tuning_configs import GridSearchConfig

from ..models.v1.tuning_models import GridSearchRequest, GridSearchResponse

from .common.result_coercion import to_payload


def compute_grid_search(req: GridSearchRequest) -> GridSearchResponse:
    """Compute a GridSearchCV-based tuning run for the given request."""

    run_cfg = RunConfig(
        data=req.data,
        split=req.split,
        scale=req.scale,
        features=req.features,
        model=req.model,
        eval=req.eval,
    )

    gs_cfg = GridSearchConfig(
        param_grid=req.param_grid,
        cv=req.cv,
        n_jobs=req.n_jobs,
        refit=req.refit,
        return_train_score=req.return_train_score,
    )

    result = bl_grid_search(run_cfg, gs_cfg)
    return GridSearchResponse(**to_payload(result))
