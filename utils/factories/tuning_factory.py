from __future__ import annotations

from shared_schemas.run_config import RunConfig
from shared_schemas.tuning_configs import (
    LearningCurveConfig,
    ValidationCurveConfig,
    GridSearchConfig,
    RandomizedSearchConfig,
)
from utils.strategies.interfaces import TuningStrategy
from utils.strategies.tuning import (
    LearningCurveRunner,
    ValidationCurveRunner,
    GridSearchRunner,
    RandomizedSearchRunner,
)


def make_learning_curve_runner(
    cfg: RunConfig,
    lc_cfg: LearningCurveConfig,
) -> TuningStrategy:
    """
    Factory for the learning-curve tuning strategy.
    """
    return LearningCurveRunner(cfg=cfg, lc=lc_cfg)


def make_validation_curve_runner(
    cfg: RunConfig,
    vc_cfg: ValidationCurveConfig,
) -> TuningStrategy:
    """
    Factory for the validation-curve tuning strategy.
    """
    return ValidationCurveRunner(cfg=cfg, vc=vc_cfg)


def make_grid_search_runner(
    cfg: RunConfig,
    gs_cfg: GridSearchConfig,
) -> TuningStrategy:
    """
    Factory for the GridSearchCV-based tuning strategy.
    """
    return GridSearchRunner(cfg=cfg, gs=gs_cfg)


def make_random_search_runner(
    cfg: RunConfig,
    rs_cfg: RandomizedSearchConfig,
) -> TuningStrategy:
    """
    Factory for the RandomizedSearchCV-based tuning strategy.
    """
    return RandomizedSearchRunner(cfg=cfg, rs=rs_cfg)
