from __future__ import annotations

from typing import ClassVar, Dict, Literal, Optional, Union

from pydantic import BaseModel

from ..choices import SVMDecisionShape, SVMKernel


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


class SVRRegressorConfig(BaseModel):
    algo: Literal["svr"] = "svr"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "svm"

    C: float = 1.0
    kernel: SVMKernel = "rbf"
    degree: int = 3
    gamma: Union[Literal["scale", "auto"], float] = "scale"
    coef0: float = 0.0
    tol: float = 1e-3
    cache_size: float = 200.0
    epsilon: float = 0.1
    shrinking: bool = True
    max_iter: int = -1
