from __future__ import annotations

from typing import Dict, Optional, Union, Literal
from pydantic import BaseModel

from .types import (
    PenaltyName, SVMKernel, SVMDecisionShape,
    TreeCriterion, TreeSplitter, MaxFeaturesName,
)

class ModelModel(BaseModel):
    # which algorithm to run
    algo: Literal["logreg", "svm", "tree", "forest", "knn"] = "logreg"

    # --- Logistic Regression ---
    C: float = 1.0
    penalty: PenaltyName = "l2"
    solver: Literal["lbfgs", "liblinear", "saga", "newton-cg", "sag"] = "lbfgs"
    max_iter: int = 1000
    class_weight: Optional[Literal["balanced"]] = None
    # only relevant for elasticnet penalty
    l1_ratio: Optional[float] = 0.5

    # --- SVM (SVC) ---
    svm_C: float = 1.0
    svm_kernel: SVMKernel = "rbf"
    svm_degree: int = 3
    svm_gamma: Union[Literal["scale", "auto"], float] = "scale"
    svm_coef0: float = 0.0
    svm_shrinking: bool = True
    svm_probability: bool = False
    svm_tol: float = 1e-3
    svm_cache_size: float = 200.0
    svm_class_weight: Optional[Union[Literal["balanced"], Dict[str, float]]] = None
    svm_max_iter: int = -1
    svm_decision_function_shape: SVMDecisionShape = "ovr"
    svm_break_ties: bool = False

    # --- Decision Tree ---
    tree_criterion: TreeCriterion = "gini"
    tree_splitter: TreeSplitter = "best"
    tree_max_depth: Optional[int] = None
    tree_min_samples_split: Union[int, float] = 2
    tree_min_samples_leaf: Union[int, float] = 1
    tree_min_weight_fraction_leaf: float = 0.0
    # sklearn accepts None | int | float | "sqrt" | "log2"
    tree_max_features: Optional[Union[int, float, MaxFeaturesName]] = None
    tree_max_leaf_nodes: Optional[int] = None
    tree_min_impurity_decrease: float = 0.0
    tree_class_weight: Optional[Literal["balanced"]] = None
    tree_ccp_alpha: float = 0.0

    # --- Random Forest ---
    forest_n_estimators: int = 100
    forest_criterion: TreeCriterion = "gini"
    forest_max_depth: Optional[int] = None
    forest_min_samples_split: Union[int, float] = 2
    forest_min_samples_leaf: Union[int, float] = 1
    forest_min_weight_fraction_leaf: float = 0.0
    forest_max_features: Optional[Union[int, float, MaxFeaturesName]] = "sqrt"
    forest_max_leaf_nodes: Optional[int] = None
    forest_min_impurity_decrease: float = 0.0
    forest_bootstrap: bool = True
    forest_oob_score: bool = False
    forest_n_jobs: Optional[int] = None
    forest_random_state: Optional[int] = None
    forest_warm_start: bool = False
    forest_class_weight: Optional[Literal["balanced", "balanced_subsample"]] = None
    forest_ccp_alpha: float = 0.0
    forest_max_samples: Optional[Union[int, float]] = None

    # --- KNN ---
    knn_n_neighbors: int = 5
    knn_weights: Literal["uniform", "distance"] = "uniform"
    knn_algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    knn_leaf_size: int = 30
    knn_p: int = 2
    knn_metric: Literal["minkowski", "euclidean", "manhattan", "chebyshev"] = "minkowski"
    knn_n_jobs: Optional[int] = None
