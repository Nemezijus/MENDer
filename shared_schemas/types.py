from __future__ import annotations
from typing import Literal, TypeAlias, Union

# Features
FeatureName: TypeAlias = Literal["none", "pca", "lda", "sfs"]

# Logistic Regression
PenaltyName: TypeAlias = Literal["l2", "l1", "elasticnet", "none"]
LogRegSolver: TypeAlias = Literal["lbfgs", "liblinear", "saga", "newton-cg", "sag"]

# Trees / Forests
TreeCriterion: TypeAlias = Literal["gini", "entropy", "log_loss"]
TreeSplitter: TypeAlias = Literal["best", "random"]
MaxFeaturesName: TypeAlias = Literal["sqrt", "log2"]  # (int|float|None are also allowed at runtime)

# Regression Trees / Forests
RegTreeCriterion: TypeAlias = Literal[
    "squared_error",
    "friedman_mse",
    "absolute_error",
    "poisson",
]

# Class weights
ClassWeightBalanced: TypeAlias = Union[Literal["balanced"], None]
ForestClassWeight: TypeAlias = Union[Literal["balanced", "balanced_subsample"], None]

# Scaling
ScaleName: TypeAlias = Literal["standard", "robust", "minmax", "maxabs", "quantile", "none"]

# Classification metrics
ClassificationMetricName: TypeAlias = Literal[
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "log_loss",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "avg_precision_macro",
]

# Regression metrics
RegressionMetricName: TypeAlias = Literal[
    "r2",
    "explained_variance",
    "mse",
    "rmse",
    "mae",
    "mape",
]

# Union used by EvalModel
MetricName: TypeAlias = Union[ClassificationMetricName, RegressionMetricName]

# SVM helpers
SVMKernel: TypeAlias = Literal["linear", "poly", "rbf", "sigmoid"]
SVMDecisionShape: TypeAlias = Literal["ovr", "ovo"]

# SVR / LinearSVR helpers
LinearSVRLoss: TypeAlias = Literal["epsilon_insensitive", "squared_epsilon_insensitive"]

# LinearSVR helpers
LinearSVRLoss: TypeAlias = Literal["epsilon_insensitive", "squared_epsilon_insensitive"]

# Linear / sparse regression helpers
CoordinateDescentSelection: TypeAlias = Literal["cyclic", "random"]

# LDA solver
LDASolver: TypeAlias = Literal["svd", "lsqr", "eigen"]

# KNN helpers
KNNWeights: TypeAlias = Literal["uniform", "distance"]
KNNAlgorithm: TypeAlias = Literal["auto", "ball_tree", "kd_tree", "brute"]
KNNMetric: TypeAlias = Literal["minkowski", "euclidean", "manhattan", "chebyshev"]

# Coordinate descent selection (ElasticNet/Lasso)
CDSelection: TypeAlias = Literal["cyclic", "random"]

# RidgeClassifier helpers
RidgeSolver: TypeAlias = Literal[
    "auto",
    "svd",
    "cholesky",
    "lsqr",
    "sparse_cg",
    "sag",
    "saga",
    "lbfgs",
]

# Coordinate descent regressors helpers
CoordinateDescentSelection: TypeAlias = Literal["cyclic", "random"]

# SGDClassifier helpers
SGDLoss: TypeAlias = Literal[
    "hinge",
    "log_loss",
    "modified_huber",
    "squared_hinge",
    "perceptron",
    # Additional losses supported by sklearn (kept for completeness / UI dropdowns)
    "squared_error",
    "huber",
    "epsilon_insensitive",
    "squared_epsilon_insensitive",
]
SGDPenalty: TypeAlias = Literal["l2", "l1", "elasticnet"]
SGDLearningRate: TypeAlias = Literal["constant", "optimal", "invscaling", "adaptive"]

# HistGradientBoostingClassifier helpers
HGBLoss: TypeAlias = Literal["log_loss"]

# Ensembles
EnsembleKind: TypeAlias = Literal["voting", "bagging", "adaboost", "xgboost"]

# Decoder / probability calibration
CalibrationMethod: TypeAlias = Literal["sigmoid", "isotonic"]

# -----------------------------
# Unsupervised / clustering
# -----------------------------

# High-level modelling task (used in UI filtering / routing)
ModelTaskName: TypeAlias = Literal["classification", "regression", "unsupervised"]

# Common internal clustering validation metrics (post-fit diagnostics)
UnsupervisedMetricName: TypeAlias = Literal[
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
]