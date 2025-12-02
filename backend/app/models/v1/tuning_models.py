from __future__ import annotations

from typing import List, Optional, Union, Any, Dict

from pydantic import BaseModel, Field

from shared_schemas.model_configs import ModelConfig
from shared_schemas.run_config import DataModel
from shared_schemas.split_configs import SplitCVModel
from shared_schemas.scale_configs import ScaleModel
from shared_schemas.feature_configs import FeaturesModel
from shared_schemas.eval_configs import EvalModel

Number = Union[int, float]
ParamValue = Union[int, float, str, bool]


class BaseTuningRequest(BaseModel):
    """
    Common core for all tuning-style requests.
    """
    data: DataModel
    split: SplitCVModel          # CV-type split config (mode="kfold" etc.)
    scale: ScaleModel
    features: FeaturesModel
    model: ModelConfig
    eval: EvalModel


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------

class LearningCurveRequest(BaseTuningRequest):
    """
    Learning curve over increasing train sizes.

    `train_sizes` can be either:
      * absolute integers (sample counts), or
      * relative fractions in (0, 1].

    If `train_sizes` is None, the backend will generate a linspace
    in (0.1, 1.0] with `n_steps` points.
    """
    train_sizes: Optional[List[Number]] = Field(
        default=None,
        description=(
            "Absolute integers or relative fractions in (0,1]. "
            "If None, use n_steps to generate a linspace."
        ),
    )
    n_steps: int = Field(
        default=5,
        ge=2,
        description="If train_sizes is None, use linspace(0.1..1.0, n_steps).",
    )
    n_jobs: int = Field(
        default=1,
        description="Passed to sklearn.learning_curve.",
    )


class LearningCurveResponse(BaseModel):
    """
    Aggregated learning-curve statistics for train/validation scores.
    """
    train_sizes: List[Optional[int]]
    train_scores_mean: List[Optional[float]]
    train_scores_std: List[Optional[float]]
    val_scores_mean: List[Optional[float]]
    val_scores_std: List[Optional[float]]


# ---------------------------------------------------------------------------
# Validation curve
# ---------------------------------------------------------------------------

class ValidationCurveRequest(BaseTuningRequest):
    """
    Validation curve over one hyperparameter.

    `param_name` is the sklearn-style parameter name, e.g. 'model__C'
    when using a Pipeline. `param_range` lists the values to explore.
    """
    param_name: str = Field(
        ...,
        description="Sklearn-style parameter name, e.g. 'model__C'.",
    )
    param_range: List[ParamValue] = Field(
        ...,
        description="Values to sweep for the parameter.",
    )
    n_jobs: int = Field(
        default=1,
        description="Passed to sklearn.validation_curve.",
    )


class ValidationCurveResponse(BaseModel):
    """
    Aggregated validation-curve statistics for train/validation scores.
    """
    param_name: str
    param_range: List[ParamValue]
    train_scores_mean: List[Optional[float]]
    train_scores_std: List[Optional[float]]
    val_scores_mean: List[Optional[float]]
    val_scores_std: List[Optional[float]]


# ---------------------------------------------------------------------------
# Grid search (GridSearchCV)
# ---------------------------------------------------------------------------

class GridSearchRequest(BaseTuningRequest):
    """
    Grid-search over a parameter grid using GridSearchCV.
    """
    param_grid: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Sklearn-style param_grid: param_name -> list of values.",
    )
    cv: int = Field(
        default=5,
        ge=2,
        description="Number of CV folds.",
    )
    n_jobs: int = Field(
        default=1,
        description="Passed to GridSearchCV.",
    )
    refit: bool = Field(
        default=True,
        description="Whether to refit on the whole dataset using best params.",
    )
    return_train_score: bool = Field(
        default=False,
        description="Whether to store train scores in cv_results_.",
    )


class GridSearchResponse(BaseModel):
    """
    Summary of a GridSearchCV run.
    """
    best_params: Dict[str, Any]
    best_score: Optional[float]
    best_index: Optional[int]
    cv_results: Dict[str, List[Any]]


# ---------------------------------------------------------------------------
# Randomized search (RandomizedSearchCV)
# ---------------------------------------------------------------------------

class RandomSearchRequest(BaseTuningRequest):
    """
    Randomized hyperparameter search using RandomizedSearchCV.
    """
    param_distributions: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Sklearn-style param_distributions: param_name -> distribution "
            "or list of values."
        ),
    )
    n_iter: int = Field(
        default=20,
        ge=1,
        description="Number of parameter settings sampled.",
    )
    cv: int = Field(
        default=5,
        ge=2,
        description="Number of CV folds.",
    )
    n_jobs: int = Field(
        default=1,
        description="Passed to RandomizedSearchCV.",
    )
    refit: bool = Field(
        default=True,
        description="Whether to refit on the whole dataset using best params.",
    )
    random_state: Optional[int] = Field(
        default=None,
        description="Random state for reproducibility (if desired).",
    )
    return_train_score: bool = Field(
        default=False,
        description="Whether to store train scores in cv_results_.",
    )


class RandomSearchResponse(BaseModel):
    """
    Summary of a RandomizedSearchCV run.
    """
    best_params: Dict[str, Any]
    best_score: Optional[float]
    best_index: Optional[int]
    cv_results: Dict[str, List[Any]]
