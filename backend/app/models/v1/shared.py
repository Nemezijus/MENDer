from typing import Optional, Union, List, Literal, Dict
from pydantic import BaseModel

FeatureName = Literal["none", "pca", "lda", "sfs"]
PenaltyName = Literal["l2", "l1", "elasticnet", "none"]
TreeCriterion = Literal["gini", "entropy", "log_loss"]
TreeSplitter = Literal["best", "random"]
MaxFeaturesName = Literal["sqrt", "log2"]

class FeaturesModel(BaseModel):
    method: FeatureName = "none"

    # PCA
    pca_n: Optional[int] = None
    pca_var: float = 0.95
    pca_whiten: bool = False

    # LDA
    lda_n: Optional[int] = None
    lda_solver: Literal["svd", "lsqr", "eigen"] = "svd"
    # sklearn supports float or "auto" (for shrinkage with 'lsqr'/'eigen')
    lda_shrinkage: Optional[Union[float, Literal["auto"]]] = None
    lda_tol: float = 1e-4
    # Optional priors (list of class priors); default None
    lda_priors: Optional[List[float]] = None

    # SFS
    sfs_k: Union[int, Literal["auto"]] = "auto"
    sfs_direction: Literal["forward", "backward"] = "backward"
    sfs_cv: int = 5
    sfs_n_jobs: Optional[int] = None

class ModelModel(BaseModel):
    algo: Literal["logreg", "svm", "tree", "forest", "knn"] = "logreg"

    # Logistic Regression
    C: float = 1.0
    penalty: PenaltyName = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000
    class_weight: Optional[Literal["balanced"]] = None
    l1_ratio: Optional[float] = None  # only for elasticnet

    # SVM (SVC)
    svm_C: float = 1.0
    svm_kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf"
    svm_degree: int = 3
    svm_gamma: Union[Literal["scale", "auto"], float] = "scale"
    svm_coef0: float = 0.0
    svm_shrinking: bool = True
    svm_probability: bool = False
    svm_tol: float = 1e-3
    svm_cache_size: float = 200.0
    svm_class_weight: Optional[Union[Literal["balanced"], Dict[str, float]]] = None
    svm_max_iter: int = -1
    svm_decision_function_shape: Literal["ovr", "ovo"] = "ovr"
    svm_break_ties: bool = False

    # DecisionTreeClassifier
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

    # RandomForestClassifier
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

    # KNeighborsClassifier
    knn_n_neighbors: int = 5
    knn_weights: Literal["uniform", "distance"] = "uniform"
    knn_algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    knn_leaf_size: int = 30
    knn_p: int = 2
    knn_metric: str = "minkowski"
    knn_n_jobs: Optional[int] = None