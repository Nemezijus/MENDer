from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from .choices import TuningKind


class LearningCurveConfig(BaseModel):
    """
    Extra knobs for learning-curve style tuning, on top of RunConfig.
    """
    # If None, we will build a linspace in [0.1, 1.0] with n_steps points.
    train_sizes: Optional[List[Union[int, float]]] = None
    n_steps: int = 5
    n_jobs: int = 1


class ValidationCurveConfig(BaseModel):
    """
    validation_curve: vary one hyperparameter and get train/val scores.
    """
    param_name: str
    param_range: List[Union[int, float, str, bool, None]]
    n_jobs: int = 1


class GridSearchConfig(BaseModel):
    """
    GridSearchCV: exhaustive search over a grid of hyperparameters.
    """
    param_grid: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Sklearn-style param_grid: param -> list of values",
    )
    cv: int = 5
    n_jobs: int = 1
    refit: bool = True
    return_train_score: bool = False


class RandomizedSearchConfig(BaseModel):
    """
    RandomizedSearchCV: random sampling from distributions of hyperparameters.
    """
    param_distributions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sklearn-style param_distributions: param -> dist or list",
    )
    n_iter: int = 20
    cv: int = 5
    n_jobs: int = 1
    refit: bool = True
    random_state: Optional[int] = None
    return_train_score: bool = False


class TuningConfig(BaseModel):
    """
    Thin wrapper if you ever want RunConfig.tuning: TuningConfig, with a 'kind'.
    """
    kind: TuningKind
    learning_curve: Optional[LearningCurveConfig] = None
    validation_curve: Optional[ValidationCurveConfig] = None
    grid_search: Optional[GridSearchConfig] = None
    random_search: Optional[RandomizedSearchConfig] = None
