from __future__ import annotations

from typing import Dict, Optional, Union, Literal, Annotated, ClassVar, TypedDict
from pydantic import BaseModel, Field

from .types import (
    PenaltyName, SVMKernel, SVMDecisionShape,
    TreeCriterion, TreeSplitter, MaxFeaturesName,
)

class ModelMeta(TypedDict):
    task: str   # or ModelTask if you define it in types.py
    family: str


def get_model_meta(model_cfg: "ModelConfig") -> ModelMeta:
    cls = model_cfg.__class__
    return {
        "task": getattr(cls, "task", "classification"),
        "family": getattr(cls, "family", "other"),
    }


def get_model_task(model_cfg: "ModelConfig") -> str:
    cls = model_cfg.__class__
    return getattr(cls, "task", "classification")

def get_model_task_by_algo(algo: str) -> str:
    mapping = {
        "logreg": LogRegConfig.task,
        "svm": SVMConfig.task,
        "tree": TreeConfig.task,
        "forest": ForestConfig.task,
        "knn": KNNConfig.task,
        "linreg": LinearRegConfig.task,
    }
    return mapping.get(algo, "classification")
# -----------------------------
# Algo-specific model configs (unprefixed fields)
# -----------------------------

class LogRegConfig(BaseModel):
    algo: Literal["logreg"] = "logreg"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "linear"

    C: float = 1.0
    penalty: PenaltyName = "l2"
    solver: Literal["lbfgs", "liblinear", "saga", "newton-cg", "sag"] = "lbfgs"
    max_iter: int = 1000
    class_weight: Optional[Literal["balanced"]] = None
    # only relevant for elasticnet penalty
    l1_ratio: Optional[float] = 0.5


class SVMConfig(BaseModel):
    algo: Literal["svm"] = "svm"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "svm"

    C: float = 1.0
    kernel: SVMKernel = "rbf"
    degree: int = 3
    gamma: Union[Literal["scale", "auto"], float] = "scale"
    coef0: float = 0.0
    shrinking: bool = True
    probability: bool = False
    tol: float = 1e-3
    cache_size: float = 200.0
    class_weight: Optional[Union[Literal["balanced"], Dict[str, float]]] = None
    max_iter: int = -1
    decision_function_shape: SVMDecisionShape = "ovr"
    break_ties: bool = False


class TreeConfig(BaseModel):
    algo: Literal["tree"] = "tree"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "forest"

    criterion: TreeCriterion = "gini"
    splitter: TreeSplitter = "best"
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    # sklearn accepts None | int | float | "sqrt" | "log2"
    max_features: Optional[Union[int, float, MaxFeaturesName]] = None
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    class_weight: Optional[Literal["balanced"]] = None
    ccp_alpha: float = 0.0


class ForestConfig(BaseModel):
    algo: Literal["forest"] = "forest"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "forest"

    n_estimators: int = 100
    criterion: TreeCriterion = "gini"
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Optional[Union[int, float, MaxFeaturesName]] = "sqrt"
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: Optional[int] = None
    random_state: Optional[int] = None
    warm_start: bool = False
    class_weight: Optional[Literal["balanced", "balanced_subsample"]] = None
    ccp_alpha: float = 0.0
    max_samples: Optional[Union[int, float]] = None


class KNNConfig(BaseModel):
    algo: Literal["knn"] = "knn"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "knn"

    n_neighbors: int = 5
    weights: Literal["uniform", "distance"] = "uniform"
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: Literal["minkowski", "euclidean", "manhattan", "chebyshev"] = "minkowski"
    n_jobs: Optional[int] = None

#------------------------------------------
#               REGRESSORS
#------------------------------------------

class LinearRegConfig(BaseModel):
    algo: Literal["linreg"] = "linreg"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: Optional[int] = None
    positive: bool = False
# -----------------------------
# Discriminated union (single source for "model")
# -----------------------------
ModelConfig = Annotated[
    Union[LogRegConfig, SVMConfig, TreeConfig, ForestConfig, KNNConfig, LinearRegConfig,],
    Field(discriminator="algo"),
]


__all__ = [
    "LogRegConfig", "SVMConfig", "TreeConfig", "ForestConfig", "KNNConfig",
    "ModelConfig", "LinearRegConfig",
]
