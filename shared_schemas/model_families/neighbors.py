from __future__ import annotations

from typing import ClassVar, Literal, Optional

from pydantic import BaseModel

from ..choices import KNNAlgorithm, KNNMetric, KNNWeights


class KNNConfig(BaseModel):
    algo: Literal["knn"] = "knn"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "neighbors"

    n_neighbors: int = 5
    weights: KNNWeights = "uniform"
    algorithm: KNNAlgorithm = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: KNNMetric = "minkowski"
    metric_params: Optional[dict] = None
    n_jobs: Optional[int] = None


class KNNRegressorConfig(BaseModel):
    algo: Literal["knnreg"] = "knnreg"

    task: ClassVar[str] = "regression"
    family: ClassVar[str] = "neighbors"

    n_neighbors: int = 5
    weights: KNNWeights = "uniform"
    algorithm: KNNAlgorithm = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: KNNMetric = "minkowski"
    metric_params: Optional[dict] = None
    n_jobs: Optional[int] = None
