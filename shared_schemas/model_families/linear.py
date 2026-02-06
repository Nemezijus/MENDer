from __future__ import annotations

from typing import ClassVar, Dict, List, Literal, Optional, Union

from pydantic import BaseModel

from ..choices import (
    ClassWeightBalanced,
    CoordinateDescentSelection,
    LinearSVRLoss,
    LogRegSolver,
    PenaltyName,
    RidgeSolver,
    SGDLearningRate,
    SGDLoss,
    SGDPenalty,
)


class LogRegConfig(BaseModel):
    algo: Literal["logreg"] = "logreg"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "linear"

    C: float = 1.0
    penalty: PenaltyName = "l2"
    solver: LogRegSolver = "lbfgs"
    max_iter: int = 1000
    class_weight: ClassWeightBalanced = None
    l1_ratio: Optional[float] = 0.5


class RidgeClassifierConfig(BaseModel):
    algo: Literal["ridge"] = "ridge"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "linear"

    alpha: float = 1.0
    fit_intercept: bool = True
    max_iter: Optional[int] = None
    tol: float = 1e-4
    solver: RidgeSolver = "auto"
    class_weight: Optional[Union[Literal["balanced"], Dict[str, float]]] = None


class SGDClassifierConfig(BaseModel):
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
    class_weight: Optional[Union[Literal["balanced"], Dict[str, float]]] = None
    warm_start: bool = False
    average: Union[bool, int] = False


class LinearRegConfig(BaseModel):
    algo: Literal["linreg"] = "linreg"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: Optional[int] = None
    positive: bool = False


class RidgeRegressorConfig(BaseModel):
    algo: Literal["ridgereg"] = "ridgereg"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    alpha: float = 1.0
    fit_intercept: bool = True
    max_iter: Optional[int] = None
    tol: float = 1e-3
    solver: RidgeSolver = "auto"


class RidgeCVRegressorConfig(BaseModel):
    algo: Literal["ridgecv"] = "ridgecv"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    alphas: List[float] = [0.1, 1.0, 10.0]
    fit_intercept: bool = True
    scoring: Optional[str] = None
    cv: Optional[int] = None
    gcv_mode: Optional[Literal["auto", "svd", "eigen"]] = None


class ElasticNetRegressorConfig(BaseModel):
    algo: Literal["enet"] = "enet"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    alpha: float = 1.0
    l1_ratio: float = 0.5
    fit_intercept: bool = True
    precompute: Union[bool, str] = False
    max_iter: int = 1000
    copy_X: bool = True
    tol: float = 1e-4
    warm_start: bool = False
    positive: bool = False
    random_state: Optional[int] = None
    selection: CoordinateDescentSelection = "cyclic"


class ElasticNetCVRegressorConfig(BaseModel):
    algo: Literal["enetcv"] = "enetcv"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    l1_ratio: List[float] = [0.1, 0.5, 0.9]
    eps: float = 0.001
    n_alphas: int = 100
    alphas: Optional[List[float]] = None
    fit_intercept: bool = True
    precompute: Union[bool, str] = "auto"
    max_iter: int = 1000
    tol: float = 1e-4
    cv: Optional[int] = None
    copy_X: bool = True
    verbose: int = 0
    n_jobs: Optional[int] = None
    positive: bool = False
    random_state: Optional[int] = None
    selection: CoordinateDescentSelection = "cyclic"


class LassoRegressorConfig(BaseModel):
    algo: Literal["lasso"] = "lasso"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    alpha: float = 1.0
    fit_intercept: bool = True
    precompute: Union[bool, str] = False
    copy_X: bool = True
    max_iter: int = 1000
    tol: float = 1e-4
    warm_start: bool = False
    positive: bool = False
    random_state: Optional[int] = None
    selection: CoordinateDescentSelection = "cyclic"


class LassoCVRegressorConfig(BaseModel):
    algo: Literal["lassocv"] = "lassocv"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    eps: float = 0.001
    n_alphas: int = 100
    alphas: Optional[List[float]] = None
    fit_intercept: bool = True
    precompute: Union[bool, str] = "auto"
    max_iter: int = 1000
    tol: float = 1e-4
    cv: Optional[int] = None
    copy_X: bool = True
    verbose: int = 0
    n_jobs: Optional[int] = None
    positive: bool = False
    random_state: Optional[int] = None
    selection: CoordinateDescentSelection = "cyclic"


class BayesianRidgeRegressorConfig(BaseModel):
    algo: Literal["bayridge"] = "bayridge"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    n_iter: int = 300
    tol: float = 1e-3
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
    alpha_init: Optional[float] = None
    lambda_init: Optional[float] = None
    compute_score: bool = False
    fit_intercept: bool = True
    copy_X: bool = True
    verbose: bool = False


class LinearSVRRegressorConfig(BaseModel):
    algo: Literal["linsvr"] = "linsvr"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "linear"

    epsilon: float = 0.0
    tol: float = 1e-4
    C: float = 1.0
    loss: LinearSVRLoss = "epsilon_insensitive"
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    dual: Union[bool, str] = "auto"
    verbose: int = 0
    random_state: Optional[int] = None
    max_iter: int = 1000
