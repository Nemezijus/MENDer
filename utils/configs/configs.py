# utils/configs.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Union, Sequence, Dict

# ---- data I/O ----
@dataclass
class DataConfig:
    x_path: Optional[str] = None
    y_path: Optional[str] = None
    npz_path: Optional[str] = None
    x_key: str = "X"
    y_key: str = "y"

# ---- split ----
@dataclass
class SplitConfig:
    train_frac: float = 0.8
    stratified: bool = True
    # ---- for cross-validation ----
    mode: Literal["holdout", "kfold"] = "holdout"   # new
    n_splits: int = 5
    shuffle: bool = True

# ---- scaling ----
ScaleName = Literal["standard", "robust", "minmax", "maxabs", "quantile", "none"]
@dataclass
class ScaleConfig:
    method: ScaleName = "standard"

# ---- features ----
FeatureName = Literal["none", "pca", "lda", "sfs"]  # later: "pls", etc.
@dataclass
class FeatureConfig:
    method: FeatureName = "none"

    # PCA
    pca_n: Optional[int] = None
    pca_var: float = 0.95
    pca_whiten: bool = False

    # LDA
    lda_n: Optional[int] = None
    lda_solver: Literal["svd", "lsqr", "eigen"] = "svd"
    lda_shrinkage: Optional[Union[float, Literal["auto"]]] = None
    lda_tol: float = 1e-4
    lda_priors: Optional[Sequence[float]] = None

    # SFS
    sfs_k: Union[int, Literal["auto"]] = "auto"
    sfs_direction: Literal["forward", "backward"] = "backward"
    sfs_cv: int = 5
    sfs_n_jobs: Optional[int] = None


# ---- model ----
PenaltyName = Literal["l2", "l1", "elasticnet", "none"]
# Common alias types for trees/forests
TreeCriterion = Literal["gini", "entropy", "log_loss"]
TreeSplitter = Literal["best", "random"]
MaxFeaturesName = Literal["sqrt", "log2"]  # None means "all features"

@dataclass
class ModelConfig:
    algo: Literal["logreg", "svm", "tree", "forest", "knn"] = "logreg"

    # Logistic Regression
    C: float = 1.0
    penalty: PenaltyName = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000
    class_weight: Optional[Literal["balanced"]] = None
    l1_ratio: Optional[float] = None  # only for elasticnet

    # --- SVM (SVC) ---
    svm_C: float = 1.0
    svm_kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf"
    svm_degree: int = 3
    svm_gamma: Literal["scale", "auto"] | float = "scale"
    svm_coef0: float = 0.0
    svm_shrinking: bool = True
    svm_probability: bool = False
    svm_tol: float = 1e-3
    svm_cache_size: float = 200.0
    svm_class_weight: Dict[str, float] | Literal["balanced", None] = None
    svm_max_iter: int = -1
    svm_decision_function_shape: Literal["ovr", "ovo"] = "ovr"
    svm_break_ties: bool = False

    # --- DecisionTreeClassifier ---
    tree_criterion: TreeCriterion = "gini"
    tree_splitter: TreeSplitter = "best"
    tree_max_depth: Optional[int] = None
    tree_min_samples_split: Union[int, float] = 2
    tree_min_samples_leaf: Union[int, float] = 1
    tree_min_weight_fraction_leaf: float = 0.0
    tree_max_features: Optional[Union[int, float, MaxFeaturesName]] = None
    tree_max_leaf_nodes: Optional[int] = None
    tree_min_impurity_decrease: float = 0.0
    tree_class_weight: Optional[Union[Literal["balanced"], Dict[str, float]]] = None
    tree_ccp_alpha: float = 0.0

    # --- RandomForestClassifier ---
    rf_n_estimators: int = 100
    rf_criterion: TreeCriterion = "gini"
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: Union[int, float] = 2
    rf_min_samples_leaf: Union[int, float] = 1
    rf_min_weight_fraction_leaf: float = 0.0
    rf_max_features: Optional[Union[int, float, MaxFeaturesName]] = "sqrt"
    rf_max_leaf_nodes: Optional[int] = None
    rf_min_impurity_decrease: float = 0.0
    rf_bootstrap: bool = True
    rf_oob_score: bool = False
    rf_n_jobs: Optional[int] = None
    rf_class_weight: Optional[Union[Literal["balanced", "balanced_subsample"], Dict[str, float]]] = None
    rf_ccp_alpha: float = 0.0
    rf_warm_start: bool = False

    # --- KNeighborsClassifier ---
    knn_n_neighbors: int = 5
    knn_weights: Literal["uniform", "distance"] = "uniform"
    knn_algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    knn_leaf_size: int = 30
    knn_p: int = 2  # 1=manhattan, 2=euclidean
    knn_metric: str = "minkowski"  # can be 'minkowski','euclidean','manhattan', etc.
    knn_n_jobs: Optional[int] = None

# ---- evaluation ----
MetricName = Literal["accuracy", "balanced_accuracy", "f1_macro"]
@dataclass
class EvalConfig:
    metric: MetricName = "accuracy"
    n_shuffles: int = 0
    seed: Optional[int] = None
    progress_id: Optional[str] = None

# ---- top-level run config ----
@dataclass
class RunConfig:
    data: DataConfig
    split: SplitConfig
    scale: ScaleConfig
    features: FeatureConfig
    model: ModelConfig
    eval: EvalConfig
