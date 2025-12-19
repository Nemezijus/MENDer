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

# LDA solver
LDASolver: TypeAlias = Literal["svd", "lsqr", "eigen"]

# KNN helpers
KNNWeights: TypeAlias = Literal["uniform", "distance"]
KNNAlgorithm: TypeAlias = Literal["auto", "ball_tree", "kd_tree", "brute"]
KNNMetric: TypeAlias = Literal["minkowski", "euclidean", "manhattan", "chebyshev"]

# Ensembles
EnsembleKind: TypeAlias = Literal["voting", "bagging", "adaboost", "xgboost"]