from __future__ import annotations

from shared_schemas.run_config import RunConfig
from shared_schemas.tuning_configs import GridSearchConfig
from utils.factories.tuning_factory import make_grid_search_runner

from ..models.v1.tuning_models import (
    GridSearchRequest,
    GridSearchResponse,
)

from .common.result_coercion import to_payload


def compute_grid_search(req: GridSearchRequest) -> GridSearchResponse:
    """
    Thin service wrapper for GridSearchCV-based tuning.

    All heavy lifting (data loading, pipeline construction, GridSearchCV)
    is handled in the business-logic layer (GridSearchRunner); this service
    only maps the API models to shared configs and back.
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
    gs_cfg = GridSearchConfig(
        param_grid=req.param_grid,
        cv=req.cv,
        n_jobs=req.n_jobs,
        refit=req.refit,
        return_train_score=req.return_train_score,
    )

    # 3) Delegate to the tuning strategy
    runner = make_grid_search_runner(run_cfg, gs_cfg)
    result = runner.run()

    # 4) Adapt to API response model
    return GridSearchResponse(**to_payload(result))
