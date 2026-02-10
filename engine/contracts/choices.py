from __future__ import annotations

"""Literal-based "choice" types used across schemas.

This module centralizes the small enumerations (TypeAlias + Literal) that are
shared across multiple schema modules.

Design intent:
- Keep this file dependency-free (stdlib + typing only).
- Prefer importing choice sets from here rather than repeating Literal[...] in
  multiple schema files.
"""

from typing import Literal, TypeAlias, Union


# -----------------------------
# Common / high-level
# -----------------------------

# High-level modelling task (used in UI filtering / routing)
ModelTaskName: TypeAlias = Literal["classification", "regression", "unsupervised"]

# Supervised-only kind (used for metrics, ensembles, etc.)
ProblemKind: TypeAlias = Literal["classification", "regression"]


# -----------------------------
# Feature selection / reduction
# -----------------------------

FeatureName: TypeAlias = Literal["none", "pca", "lda", "sfs"]

SFSK: TypeAlias = Union[int, Literal["auto"]]
SFSDirection: TypeAlias = Literal["forward", "backward"]


# -----------------------------
# Scaling
# -----------------------------

ScaleName: TypeAlias = Literal["standard", "robust", "minmax", "maxabs", "quantile", "none"]


# -----------------------------
# Logistic Regression
# -----------------------------

PenaltyName: TypeAlias = Literal["l2", "l1", "elasticnet", "none"]
LogRegSolver: TypeAlias = Literal["lbfgs", "liblinear", "saga", "newton-cg", "sag"]


# -----------------------------
# Trees / Forests
# -----------------------------

TreeCriterion: TypeAlias = Literal["gini", "entropy", "log_loss"]
TreeSplitter: TypeAlias = Literal["best", "random"]
MaxFeaturesName: TypeAlias = Literal["sqrt", "log2"]  # (int|float|None are also allowed at runtime)

RegTreeCriterion: TypeAlias = Literal[
    "squared_error",
    "friedman_mse",
    "absolute_error",
    "poisson",
]


# -----------------------------
# Class weights
# -----------------------------

ClassWeightBalanced: TypeAlias = Union[Literal["balanced"], None]
ForestClassWeight: TypeAlias = Union[Literal["balanced", "balanced_subsample"], None]


# -----------------------------
# Metrics
# -----------------------------

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

RegressionMetricName: TypeAlias = Literal[
    "r2",
    "explained_variance",
    "mse",
    "rmse",
    "mae",
    "mape",
]

UnsupervisedMetricName: TypeAlias = Literal[
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
]

MetricName: TypeAlias = Union[
    ClassificationMetricName,
    RegressionMetricName,
    UnsupervisedMetricName,
]


# -----------------------------
# SVM helpers
# -----------------------------

SVMKernel: TypeAlias = Literal["linear", "poly", "rbf", "sigmoid"]
SVMDecisionShape: TypeAlias = Literal["ovr", "ovo"]

LinearSVRLoss: TypeAlias = Literal["epsilon_insensitive", "squared_epsilon_insensitive"]


# -----------------------------
# Linear / sparse regression helpers
# -----------------------------

CoordinateDescentSelection: TypeAlias = Literal["cyclic", "random"]


# -----------------------------
# LDA solver
# -----------------------------

LDASolver: TypeAlias = Literal["svd", "lsqr", "eigen"]


# -----------------------------
# KNN helpers
# -----------------------------

KNNWeights: TypeAlias = Literal["uniform", "distance"]
KNNAlgorithm: TypeAlias = Literal["auto", "ball_tree", "kd_tree", "brute"]
KNNMetric: TypeAlias = Literal["minkowski", "euclidean", "manhattan", "chebyshev"]


# -----------------------------
# RidgeClassifier helpers
# -----------------------------

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


# -----------------------------
# SGDClassifier helpers
# -----------------------------

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


# -----------------------------
# HistGradientBoostingClassifier helpers
# -----------------------------

HGBLoss: TypeAlias = Literal["log_loss"]


# -----------------------------
# Ensembles
# -----------------------------

EnsembleKind: TypeAlias = Literal["voting", "bagging", "adaboost", "xgboost"]


# -----------------------------
# Decoder / probability calibration
# -----------------------------

CalibrationMethod: TypeAlias = Literal["sigmoid", "isotonic"]


# -----------------------------
# Tuning / search
# -----------------------------

TuningKind: TypeAlias = Literal["learning_curve", "validation_curve", "grid_search", "random_search"]


# -----------------------------
# Unsupervised
# -----------------------------

FitScopeName: TypeAlias = Literal["train_only", "train_and_predict"]


__all__ = [
    # high-level
    "ModelTaskName",
    "ProblemKind",
    # features
    "FeatureName",
    "SFSK",
    "SFSDirection",
    # scaling
    "ScaleName",
    # logreg
    "PenaltyName",
    "LogRegSolver",
    # trees
    "TreeCriterion",
    "TreeSplitter",
    "MaxFeaturesName",
    "RegTreeCriterion",
    # class weights
    "ClassWeightBalanced",
    "ForestClassWeight",
    # metrics
    "ClassificationMetricName",
    "RegressionMetricName",
    "UnsupervisedMetricName",
    "MetricName",
    # svm
    "SVMKernel",
    "SVMDecisionShape",
    "LinearSVRLoss",
    # linear helpers
    "CoordinateDescentSelection",
    # lda
    "LDASolver",
    # knn
    "KNNWeights",
    "KNNAlgorithm",
    "KNNMetric",
    # ridge
    "RidgeSolver",
    # sgd
    "SGDLoss",
    "SGDPenalty",
    "SGDLearningRate",
    # hgb
    "HGBLoss",
    # ensembles
    "EnsembleKind",
    # calibration
    "CalibrationMethod",
    # tuning
    "TuningKind",
    # unsupervised
    "FitScopeName",
]
