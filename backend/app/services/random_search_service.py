# backend/app/services/random_search_service.py
from __future__ import annotations

from shared_schemas.run_config import RunConfig
from shared_schemas.tuning_configs import RandomizedSearchConfig
from utils.factories.tuning_factory import make_random_search_runner

from ..models.v1.tuning_models import (
    RandomSearchRequest,
    RandomSearchResponse,
)


def compute_random_search(req: RandomSearchRequest) -> RandomSearchResponse:
    """
    Thin service wrapper for RandomizedSearchCV-based tuning.

    All heavy lifting (data loading, pipeline construction, RandomizedSearchCV)
    is handled in the business-logic layer (RandomizedSearchRunner); this
    service only maps the API models to shared configs and back.
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
    rs_cfg = RandomizedSearchConfig(
        param_distributions=req.param_distributions,
        n_iter=req.n_iter,
        cv=req.cv,
        n_jobs=req.n_jobs,
        refit=req.refit,
        random_state=req.random_state,
        return_train_score=req.return_train_score,
    )

    # 3) Delegate to the tuning strategy
    runner = make_random_search_runner(run_cfg, rs_cfg)
    result_dict = runner.run()

    # 4) Adapt to API response model
    return RandomSearchResponse(**result_dict)
