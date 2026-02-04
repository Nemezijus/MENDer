from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from .common import ResultModel, JSONDict

Number = Union[int, float]
ParamValue = Union[int, float, str, bool]


class LearningCurveResult(ResultModel):
    metric_used: Optional[str] = None
    note: Optional[str] = None

    train_sizes: List[Optional[int]] = Field(default_factory=list)
    train_scores_mean: List[Optional[float]] = Field(default_factory=list)
    train_scores_std: List[Optional[float]] = Field(default_factory=list)
    val_scores_mean: List[Optional[float]] = Field(default_factory=list)
    val_scores_std: List[Optional[float]] = Field(default_factory=list)


class ValidationCurveResult(ResultModel):
    metric_used: Optional[str] = None
    note: Optional[str] = None

    param_name: str
    param_range: List[ParamValue] = Field(default_factory=list)

    train_scores_mean: List[Optional[float]] = Field(default_factory=list)
    train_scores_std: List[Optional[float]] = Field(default_factory=list)
    val_scores_mean: List[Optional[float]] = Field(default_factory=list)
    val_scores_std: List[Optional[float]] = Field(default_factory=list)


class GridSearchResult(ResultModel):
    metric_used: Optional[str] = None
    note: Optional[str] = None

    best_params: JSONDict = Field(default_factory=dict)
    best_score: Optional[float] = None
    best_index: Optional[int] = None
    cv_results: Dict[str, List[Any]] = Field(default_factory=dict)


class RandomSearchResult(ResultModel):
    metric_used: Optional[str] = None
    note: Optional[str] = None

    best_params: JSONDict = Field(default_factory=dict)
    best_score: Optional[float] = None
    best_index: Optional[int] = None
    cv_results: Dict[str, List[Any]] = Field(default_factory=dict)


TuningResult = Union[LearningCurveResult, ValidationCurveResult, GridSearchResult, RandomSearchResult]
