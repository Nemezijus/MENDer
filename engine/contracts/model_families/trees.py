from __future__ import annotations

from typing import ClassVar, Dict, Literal, Optional, Union

from pydantic import BaseModel

from ..choices import (
    ForestClassWeight,
    HGBLoss,
    MaxFeaturesName,
    RegTreeCriterion,
    TreeCriterion,
    TreeSplitter,
)


class TreeConfig(BaseModel):
    algo: Literal["tree"] = "tree"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "trees"

    criterion: TreeCriterion = "gini"
    splitter: TreeSplitter = "best"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Optional[Union[int, float, MaxFeaturesName]] = None
    random_state: Optional[int] = None
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    class_weight: Optional[Dict[str, float]] = None


class ForestConfig(BaseModel):
    algo: Literal["forest"] = "forest"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "trees"

    n_estimators: int = 100
    criterion: TreeCriterion = "gini"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[int, float, MaxFeaturesName] = "sqrt"
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: Optional[int] = None
    random_state: Optional[int] = None
    verbose: int = 0
    warm_start: bool = False
    class_weight: ForestClassWeight = None
    ccp_alpha: float = 0.0
    max_samples: Optional[Union[int, float]] = None


class ExtraTreesConfig(BaseModel):
    algo: Literal["extratrees"] = "extratrees"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "trees"

    n_estimators: int = 100
    criterion: TreeCriterion = "gini"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[int, float, MaxFeaturesName] = "sqrt"
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = False
    oob_score: bool = False
    n_jobs: Optional[int] = None
    random_state: Optional[int] = None
    verbose: int = 0
    warm_start: bool = False
    class_weight: ForestClassWeight = None
    ccp_alpha: float = 0.0
    max_samples: Optional[Union[int, float]] = None


class HistGradientBoostingConfig(BaseModel):
    algo: Literal["hgb"] = "hgb"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "trees"

    loss: HGBLoss = "log_loss"
    learning_rate: float = 0.1
    max_iter: int = 100
    max_leaf_nodes: int = 31
    max_depth: Optional[int] = None
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    max_features: float = 1.0
    max_bins: int = 255
    categorical_features: Optional[list[int]] = None
    monotonic_cst: Optional[list[int]] = None
    interaction_cst: Optional[list[int]] = None
    random_state: Optional[int] = None
    warm_start: bool = False
    early_stopping: Union[bool, str] = "auto"
    scoring: Optional[str] = None
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    tol: float = 1e-7


class DecisionTreeRegressorConfig(BaseModel):
    algo: Literal["treereg"] = "treereg"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "trees"

    criterion: RegTreeCriterion = "squared_error"
    splitter: TreeSplitter = "best"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Optional[Union[int, float, MaxFeaturesName]] = None
    random_state: Optional[int] = None
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0


class RandomForestRegressorConfig(BaseModel):
    algo: Literal["rfreg"] = "rfreg"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "trees"

    n_estimators: int = 100
    criterion: RegTreeCriterion = "squared_error"
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[int, float, MaxFeaturesName] = 1.0
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: Optional[int] = None
    random_state: Optional[int] = None
    verbose: int = 0
    warm_start: bool = False
    ccp_alpha: float = 0.0
    max_samples: Optional[Union[int, float]] = None
