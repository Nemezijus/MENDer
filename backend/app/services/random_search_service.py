"""Backend tuning service: random search.

Segment 12B: delegate orchestration to the Engine façade.
"""

from __future__ import annotations

from engine.use_cases.facade import random_search as bl_random_search

from engine.contracts.run_config import RunConfig
from engine.contracts.tuning_configs import RandomizedSearchConfig

from ..models.v1.tuning_models import RandomSearchRequest, RandomSearchResponse

from .common.result_coercion import to_payload


def compute_random_search(req: RandomSearchRequest) -> RandomSearchResponse:
    """Compute a RandomizedSearchCV-based tuning run for the given request."""

    run_cfg = RunConfig(
        data=req.data,
        split=req.split,
        scale=req.scale,
        features=req.features,
        model=req.model,
        eval=req.eval,
    )

    rs_cfg = RandomizedSearchConfig(
        param_distributions=req.param_distributions,
        n_iter=req.n_iter,
        cv=req.cv,
        n_jobs=req.n_jobs,
        refit=req.refit,
        random_state=req.random_state,
        return_train_score=req.return_train_score,
    )

    result = bl_random_search(run_cfg, rs_cfg)
    return RandomSearchResponse(**to_payload(result))
