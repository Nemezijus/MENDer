from __future__ import annotations

from typing import Literal, TypeAlias

# Features
FeatureName: TypeAlias = Literal["none", "pca", "lda", "sfs"]

# Logistic Regression
PenaltyName: TypeAlias = Literal["l2", "l1", "elasticnet", "none"]

# Trees / Forests
TreeCriterion: TypeAlias = Literal["gini", "entropy", "log_loss"]
TreeSplitter: TypeAlias = Literal["best", "random"]
MaxFeaturesName: TypeAlias = Literal["sqrt", "log2"]  # note: int|float|None are also allowed at runtime

# Scaling (keep your previous list)
ScaleName: TypeAlias = Literal["standard", "robust", "minmax", "maxabs", "quantile", "none"]

# Metrics (central place; add more later)
MetricName: TypeAlias = Literal["accuracy", "balanced_accuracy", "f1_macro"]

# SVM helpers (optional, but nice to centralize)
SVMKernel: TypeAlias = Literal["linear", "poly", "rbf", "sigmoid"]
SVMDecisionShape: TypeAlias = Literal["ovr", "ovo"]

# LDA solver
LDASolver: TypeAlias = Literal["svd", "lsqr", "eigen"]
