from __future__ import annotations

from typing import ClassVar, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class KMeansConfig(BaseModel):
    algo: Literal["kmeans"] = "kmeans"

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "unsupervised"

    n_clusters: int = 2
    init: Union[Literal["k-means++", "random"], str] = "k-means++"
    n_init: Union[int, Literal["auto"]] = "auto"
    max_iter: int = 300
    tol: float = 1e-4
    verbose: int = 0
    random_state: Optional[int] = None


class DBSCANConfig(BaseModel):
    algo: Literal["dbscan"] = "dbscan"

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "unsupervised"

    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"
    metric_params: Optional[dict] = None
    algorithm: str = "auto"
    leaf_size: int = 30
    p: Optional[int] = None
    n_jobs: Optional[int] = None


class SpectralClusteringConfig(BaseModel):
    algo: Literal["spectral"] = "spectral"

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "unsupervised"

    n_clusters: int = 2
    eigen_solver: Optional[str] = None
    n_components: int = 2
    random_state: Optional[int] = None
    n_init: int = 10
    gamma: float = 1.0
    affinity: str = "rbf"
    n_neighbors: int = 10
    eigen_tol: float = 0.0
    assign_labels: str = "kmeans"
    degree: float = 3.0
    coef0: float = 1.0
    kernel_params: Optional[dict] = None
    n_jobs: Optional[int] = None


class AgglomerativeClusteringConfig(BaseModel):
    algo: Literal["agglo"] = "agglo"

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "unsupervised"

    n_clusters: Optional[int] = 2
    metric: str = "euclidean"
    linkage: str = "ward"
    distance_threshold: Optional[float] = None
    compute_full_tree: Union[bool, Literal["auto"]] = "auto"
    compute_distances: bool = False


class GaussianMixtureConfig(BaseModel):
    algo: Literal["gmm"] = "gmm"

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "unsupervised"

    n_components: int = 2
    covariance_type: str = "full"
    tol: float = 1e-3
    reg_covar: float = 1e-6
    max_iter: int = 100
    n_init: int = 1
    init_params: str = "kmeans"
    weights_init: Optional[list[float]] = None
    means_init: Optional[list[float]] = None
    precisions_init: Optional[list[float]] = None
    random_state: Optional[int] = None
    warm_start: bool = False


class BayesianGaussianMixtureConfig(BaseModel):
    algo: Literal["bgmm"] = "bgmm"

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "unsupervised"

    n_components: int = 2
    covariance_type: str = "full"
    tol: float = 1e-3
    reg_covar: float = 1e-6
    max_iter: int = 100
    n_init: int = 1
    init_params: str = "kmeans"
    weight_concentration_prior_type: str = "dirichlet_process"
    weight_concentration_prior: Optional[float] = None
    mean_precision_prior: Optional[float] = None
    mean_prior: Optional[list[float]] = None
    degrees_of_freedom_prior: Optional[float] = None
    covariance_prior: Optional[list[float]] = None
    random_state: Optional[int] = None
    warm_start: bool = False


class MeanShiftConfig(BaseModel):
    algo: Literal["meanshift"] = "meanshift"

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "unsupervised"

    bandwidth: Optional[float] = None
    seeds: Optional[list[list[float]]] = None
    bin_seeding: bool = False
    min_bin_freq: int = 1
    cluster_all: bool = True
    n_jobs: Optional[int] = None
    max_iter: int = 300


class BirchConfig(BaseModel):
    algo: Literal["birch"] = "birch"

    # Allow both internal `copy_` and external payload field name `copy`.
    model_config = ConfigDict(populate_by_name=True)

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "unsupervised"

    threshold: float = 0.5
    branching_factor: int = 50
    n_clusters: Optional[int] = 2
    compute_labels: bool = True
    copy_: bool = Field(True, alias="copy")
