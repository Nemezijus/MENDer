# shared_schemas/model_configs.py
from __future__ import annotations

from typing import Dict, Optional, Union, Literal, Annotated, ClassVar, TypedDict, List
from pydantic import BaseModel, Field

from .types import (
    PenaltyName, SVMKernel, SVMDecisionShape,
    TreeCriterion, TreeSplitter, MaxFeaturesName,
    RidgeSolver, SGDLoss, SGDPenalty, SGDLearningRate, HGBLoss,
    RegTreeCriterion, CoordinateDescentSelection, LinearSVRLoss,
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
        "gnb": GaussianNBConfig.task,
        "ridge": RidgeClassifierConfig.task,
        "sgd": SGDClassifierConfig.task,
        "extratrees": ExtraTreesConfig.task,
        "hgb": HistGradientBoostingConfig.task,
        "linreg": LinearRegConfig.task,

        # regressors
        "ridgereg": RidgeRegressorConfig.task,
        "ridgecv": RidgeCVRegressorConfig.task,
        "enet": ElasticNetRegressorConfig.task,
        "enetcv": ElasticNetCVRegressorConfig.task,
        "lasso": LassoRegressorConfig.task,
        "lassocv": LassoCVRegressorConfig.task,
        "bayridge": BayesianRidgeRegressorConfig.task,
        "svr": SVRRegressorConfig.task,
        "linsvr": LinearSVRRegressorConfig.task,
        "knnreg": KNNRegressorConfig.task,
        "treereg": DecisionTreeRegressorConfig.task,
        "rfreg": RandomForestRegressorConfig.task,
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


class GaussianNBConfig(BaseModel):
    """Gaussian Naive Bayes (continuous features)."""

    algo: Literal["gnb"] = "gnb"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "nb"

    priors: Optional[List[float]] = None
    var_smoothing: float = 1e-9


class RidgeClassifierConfig(BaseModel):
    """RidgeClassifier (linear baseline with L2 regularization)."""

    algo: Literal["ridge"] = "ridge"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "linear"

    alpha: float = 1.0
    fit_intercept: bool = True
    class_weight: Optional[Literal["balanced"]] = None
    solver: RidgeSolver = "auto"
    max_iter: Optional[int] = None
    tol: float = 1e-4


class SGDClassifierConfig(BaseModel):
    """SGDClassifier (fast linear model, good for large/sparse problems)."""

    algo: Literal["sgd"] = "sgd"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "linear"

    loss: SGDLoss = "hinge"
    penalty: SGDPenalty = "l2"
    alpha: float = 0.0001
    l1_ratio: float = 0.15
    fit_intercept: bool = True

    max_iter: int = 1000
    tol: float = 1e-3
    shuffle: bool = True
    verbose: int = 0

    epsilon: float = 0.1
    n_jobs: Optional[int] = None

    learning_rate: SGDLearningRate = "optimal"
    eta0: float = 0.0
    power_t: float = 0.5

    early_stopping: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 5

    class_weight: Optional[Literal["balanced"]] = None
    warm_start: bool = False
    average: Union[bool, int] = False


class ExtraTreesConfig(BaseModel):
    """ExtraTreesClassifier (Extremely Randomized Trees)."""

    algo: Literal["extratrees"] = "extratrees"

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
    bootstrap: bool = False
    oob_score: bool = False
    n_jobs: Optional[int] = None
    random_state: Optional[int] = None
    warm_start: bool = False
    class_weight: Optional[Literal["balanced", "balanced_subsample"]] = None
    ccp_alpha: float = 0.0
    max_samples: Optional[Union[int, float]] = None


class HistGradientBoostingConfig(BaseModel):
    """HistGradientBoostingClassifier (modern sklearn boosting for tabular data)."""

    algo: Literal["hgb"] = "hgb"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "boosting"

    loss: HGBLoss = "log_loss"
    learning_rate: float = 0.1

    max_iter: int = 100
    max_leaf_nodes: int = 31
    max_depth: Optional[int] = None
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    max_features: float = 1.0
    max_bins: int = 255

    # early stopping related
    early_stopping: Union[bool, Literal["auto"]] = "auto"
    scoring: Optional[str] = "loss"
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    tol: float = 1e-7

    verbose: int = 0
    random_state: Optional[int] = None
    class_weight: Optional[Literal["balanced"]] = None

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


class RidgeRegressorConfig(BaseModel):
    """Ridge regression (linear model with L2 regularization)."""

    algo: Literal["ridgereg"] = "ridgereg"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    alpha: float = 1.0
    fit_intercept: bool = True
    solver: RidgeSolver = "auto"
    max_iter: Optional[int] = None
    tol: float = 1e-4
    random_state: Optional[int] = None
    positive: bool = False


class RidgeCVRegressorConfig(BaseModel):
    """Ridge regression with built-in cross-validation over alphas."""

    algo: Literal["ridgecv"] = "ridgecv"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    alphas: List[float] = Field(default_factory=lambda: [0.1, 1.0, 10.0])
    fit_intercept: bool = True
    scoring: Optional[str] = None
    # None = leave to sklearn default (often efficient GCV when possible)
    cv: Optional[int] = None
    gcv_mode: Optional[Literal["auto", "svd", "eigen"]] = None
    alpha_per_target: bool = False


class ElasticNetRegressorConfig(BaseModel):
    """ElasticNet regression (L1 + L2)."""

    algo: Literal["enet"] = "enet"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    alpha: float = 1.0
    l1_ratio: float = 0.5
    fit_intercept: bool = True
    max_iter: int = 1000
    tol: float = 1e-4
    selection: CoordinateDescentSelection = "cyclic"
    random_state: Optional[int] = None
    positive: bool = False


class ElasticNetCVRegressorConfig(BaseModel):
    """ElasticNet with built-in cross-validation."""

    algo: Literal["enetcv"] = "enetcv"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    l1_ratio: List[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    eps: float = 1e-3
    n_alphas: int = 100
    fit_intercept: bool = True
    max_iter: int = 1000
    tol: float = 1e-4
    cv: int = 5
    n_jobs: Optional[int] = None
    selection: CoordinateDescentSelection = "cyclic"
    random_state: Optional[int] = None
    positive: bool = False


class LassoRegressorConfig(BaseModel):
    """Lasso regression (L1)."""

    algo: Literal["lasso"] = "lasso"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    alpha: float = 1.0
    fit_intercept: bool = True
    max_iter: int = 1000
    tol: float = 1e-4
    selection: CoordinateDescentSelection = "cyclic"
    random_state: Optional[int] = None
    positive: bool = False


class LassoCVRegressorConfig(BaseModel):
    """Lasso regression with built-in cross-validation."""

    algo: Literal["lassocv"] = "lassocv"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    eps: float = 1e-3
    n_alphas: int = 100
    fit_intercept: bool = True
    max_iter: int = 1000
    tol: float = 1e-4
    cv: int = 5
    n_jobs: Optional[int] = None
    selection: CoordinateDescentSelection = "cyclic"
    random_state: Optional[int] = None
    positive: bool = False


class BayesianRidgeRegressorConfig(BaseModel):
    """Bayesian ridge regression."""

    algo: Literal["bayridge"] = "bayridge"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    n_iter: int = 300
    tol: float = 1e-3
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
    compute_score: bool = False
    fit_intercept: bool = True
    copy_X: bool = True
    verbose: bool = False


class SVRRegressorConfig(BaseModel):
    """Support Vector Regression (kernel SVR)."""

    algo: Literal["svr"] = "svr"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "svm"

    C: float = 1.0
    kernel: SVMKernel = "rbf"
    degree: int = 3
    gamma: Union[Literal["scale", "auto"], float] = "scale"
    coef0: float = 0.0
    shrinking: bool = True
    tol: float = 1e-3
    cache_size: float = 200.0
    max_iter: int = -1
    epsilon: float = 0.1


class LinearSVRRegressorConfig(BaseModel):
    """Linear Support Vector Regression (faster than kernel SVR)."""

    algo: Literal["linsvr"] = "linsvr"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "svm"

    C: float = 1.0
    loss: LinearSVRLoss = "epsilon_insensitive"
    epsilon: float = 0.0
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    dual: bool = True
    tol: float = 1e-4
    max_iter: int = 1000
    random_state: Optional[int] = None


class KNNRegressorConfig(BaseModel):
    """KNeighborsRegressor."""

    algo: Literal["knnreg"] = "knnreg"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "knn"

    n_neighbors: int = 5
    weights: Literal["uniform", "distance"] = "uniform"
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: Literal["minkowski", "euclidean", "manhattan", "chebyshev"] = "minkowski"
    n_jobs: Optional[int] = None


class DecisionTreeRegressorConfig(BaseModel):
    """DecisionTreeRegressor."""

    algo: Literal["treereg"] = "treereg"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "forest"

    criterion: RegTreeCriterion = "squared_error"
    splitter: TreeSplitter = "best"
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Optional[Union[int, float, MaxFeaturesName]] = None
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    random_state: Optional[int] = None
    ccp_alpha: float = 0.0


class RandomForestRegressorConfig(BaseModel):
    """RandomForestRegressor (ensemble of decision trees)."""

    algo: Literal["rfreg"] = "rfreg"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "forest"

    n_estimators: int = 100
    criterion: RegTreeCriterion = "squared_error"
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Optional[Union[int, float, MaxFeaturesName]] = 1.0
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: Optional[int] = None
    random_state: Optional[int] = None
    warm_start: bool = False
    ccp_alpha: float = 0.0
    max_samples: Optional[Union[int, float]] = None
# -----------------------------
# Discriminated union (single source for "model")
# -----------------------------
ModelConfig = Annotated[
    Union[
        LogRegConfig,
        SVMConfig,
        TreeConfig,
        ForestConfig,
        KNNConfig,
        GaussianNBConfig,
        RidgeClassifierConfig,
        SGDClassifierConfig,
        ExtraTreesConfig,
        HistGradientBoostingConfig,
        LinearRegConfig,

        RidgeRegressorConfig,
        RidgeCVRegressorConfig,
        ElasticNetRegressorConfig,
        ElasticNetCVRegressorConfig,
        LassoRegressorConfig,
        LassoCVRegressorConfig,
        BayesianRidgeRegressorConfig,
        SVRRegressorConfig,
        LinearSVRRegressorConfig,
        KNNRegressorConfig,
        DecisionTreeRegressorConfig,
        RandomForestRegressorConfig,
    ],
    Field(discriminator="algo"),
]


__all__ = [
    "LogRegConfig",
    "SVMConfig",
    "TreeConfig",
    "ForestConfig",
    "KNNConfig",
    "GaussianNBConfig",
    "RidgeClassifierConfig",
    "SGDClassifierConfig",
    "ExtraTreesConfig",
    "HistGradientBoostingConfig",
    "LinearRegConfig",

    "RidgeRegressorConfig",
    "RidgeCVRegressorConfig",
    "ElasticNetRegressorConfig",
    "ElasticNetCVRegressorConfig",
    "LassoRegressorConfig",
    "LassoCVRegressorConfig",
    "BayesianRidgeRegressorConfig",
    "SVRRegressorConfig",
    "LinearSVRRegressorConfig",
    "KNNRegressorConfig",
    "DecisionTreeRegressorConfig",
    "RandomForestRegressorConfig",
    "ModelConfig",
]
