from __future__ import annotations

from typing import Dict, Optional, Union, Literal, Annotated, ClassVar, TypedDict, List
from pydantic import BaseModel, Field

from .choices import (
    ClassWeightBalanced,
    CoordinateDescentSelection,
    ForestClassWeight,
    HGBLoss,
    KNNAlgorithm,
    KNNMetric,
    KNNWeights,
    LinearSVRLoss,
    LogRegSolver,
    MaxFeaturesName,
    PenaltyName,
    RegTreeCriterion,
    RidgeSolver,
    SGDLearningRate,
    SGDLoss,
    SGDPenalty,
    SVMDecisionShape,
    SVMKernel,
    TreeCriterion,
    TreeSplitter,
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

        # clustering / unsupervised
        "kmeans": KMeansConfig.task,
        "dbscan": DBSCANConfig.task,
        "spectral": SpectralClusteringConfig.task,
        "agglo": AgglomerativeClusteringConfig.task,
        "gmm": GaussianMixtureConfig.task,
        "bgmm": BayesianGaussianMixtureConfig.task,
        "meanshift": MeanShiftConfig.task,
        "birch": BirchConfig.task,
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
    solver: LogRegSolver = "lbfgs"
    max_iter: int = 1000
    class_weight: ClassWeightBalanced = None
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
    class_weight: ClassWeightBalanced = None
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
    class_weight: ForestClassWeight = None
    ccp_alpha: float = 0.0
    max_samples: Optional[Union[int, float]] = None


class KNNConfig(BaseModel):
    algo: Literal["knn"] = "knn"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "knn"

    n_neighbors: int = 5
    weights: KNNWeights = "uniform"
    algorithm: KNNAlgorithm = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: KNNMetric = "minkowski"
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
    class_weight: ClassWeightBalanced = None
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

    class_weight: ClassWeightBalanced = None
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
    class_weight: ForestClassWeight = None
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
    class_weight: ClassWeightBalanced = None

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
    weights: KNNWeights = "uniform"
    algorithm: KNNAlgorithm = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: KNNMetric = "minkowski"
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


#------------------------------------------
#          CLUSTERING / UNSUPERVISED
#------------------------------------------

class KMeansConfig(BaseModel):
    """KMeans clustering."""

    algo: Literal["kmeans"] = "kmeans"

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "unsupervised"

    n_clusters: int = 2
    init: Literal["k-means++", "random"] = "k-means++"
    n_init: Union[Literal["auto"], int] = "auto"
    max_iter: int = 300
    tol: float = 1e-4
    verbose: int = 0
    random_state: Optional[int] = None
    algorithm: Union[Literal["lloyd", "elkan", "auto"], str] = "lloyd"


class DBSCANConfig(BaseModel):
    """DBSCAN density-based clustering."""

    algo: Literal["dbscan"] = "dbscan"

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "unsupervised"

    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"
    metric_params: Optional[Dict[str, float]] = None
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = 30
    p: Optional[float] = None
    n_jobs: Optional[int] = None


class SpectralClusteringConfig(BaseModel):
    """SpectralClustering (graph-based clustering)."""

    algo: Literal["spectral"] = "spectral"

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "unsupervised"

    n_clusters: int = 2
    eigen_solver: Optional[Literal["arpack", "lobpcg", "amg"]] = None
    n_components: Optional[int] = None
    random_state: Optional[int] = None
    n_init: int = 10
    gamma: float = 1.0
    affinity: Literal[
        "nearest_neighbors",
        "rbf",
        "precomputed",
        "precomputed_nearest_neighbors",
    ] = "rbf"
    n_neighbors: int = 10
    assign_labels: Literal["kmeans", "discretize", "cluster_qr"] = "kmeans"
    degree: int = 3
    coef0: float = 1.0


class AgglomerativeClusteringConfig(BaseModel):
    """Agglomerative hierarchical clustering."""

    algo: Literal["agglo"] = "agglo"

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "unsupervised"

    n_clusters: Optional[int] = 2
    metric: str = "euclidean"
    linkage: Literal["ward", "complete", "average", "single"] = "ward"
    distance_threshold: Optional[float] = None
    compute_full_tree: Union[bool, Literal["auto"]] = "auto"
    # Required for dendrogram rendering (children_ + distances_).
    # Enabled by default; can be turned off for very large problems.
    compute_distances: bool = True


class GaussianMixtureConfig(BaseModel):
    """Gaussian Mixture Model (GMM)."""

    algo: Literal["gmm"] = "gmm"

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "mixture"

    n_components: int = 2
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    tol: float = 1e-3
    reg_covar: float = 1e-6
    max_iter: int = 100
    n_init: int = 1
    init_params: Literal["kmeans", "k-means++", "random", "random_from_data"] = "kmeans"
    weights_init: Optional[List[float]] = None
    means_init: Optional[List[List[float]]] = None
    random_state: Optional[int] = None
    warm_start: bool = False
    verbose: int = 0


class BayesianGaussianMixtureConfig(BaseModel):
    """Variational Bayesian Gaussian mixture (BGMM)."""

    algo: Literal["bgmm"] = "bgmm"

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "mixture"

    n_components: int = 2
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    tol: float = 1e-3
    reg_covar: float = 1e-6
    max_iter: int = 100
    n_init: int = 1
    init_params: Literal["kmeans", "k-means++", "random", "random_from_data"] = "kmeans"
    weight_concentration_prior_type: Literal[
        "dirichlet_process",
        "dirichlet_distribution",
    ] = "dirichlet_process"
    weight_concentration_prior: Optional[float] = None
    mean_precision_prior: Optional[float] = None
    degrees_of_freedom_prior: Optional[float] = None
    random_state: Optional[int] = None
    warm_start: bool = False
    verbose: int = 0


class MeanShiftConfig(BaseModel):
    """MeanShift clustering."""

    algo: Literal["meanshift"] = "meanshift"

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "unsupervised"

    bandwidth: Optional[float] = None
    bin_seeding: bool = False
    min_bin_freq: int = 1
    cluster_all: bool = True
    n_jobs: Optional[int] = None


class BirchConfig(BaseModel):
    """Birch clustering."""

    algo: Literal["birch"] = "birch"

    task: ClassVar[str] = "unsupervised"
    family: ClassVar[str] = "unsupervised"

    threshold: float = 0.5
    branching_factor: int = 50
    n_clusters: Optional[int] = 2
    compute_labels: bool = True
    copy: bool = True



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

        KMeansConfig,
        DBSCANConfig,
        SpectralClusteringConfig,
        AgglomerativeClusteringConfig,
        GaussianMixtureConfig,
        BayesianGaussianMixtureConfig,
        MeanShiftConfig,
        BirchConfig,
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

    "KMeansConfig",
    "DBSCANConfig",
    "SpectralClusteringConfig",
    "AgglomerativeClusteringConfig",
    "GaussianMixtureConfig",
    "BayesianGaussianMixtureConfig",
    "MeanShiftConfig",
    "BirchConfig",
    "ModelConfig",
]
