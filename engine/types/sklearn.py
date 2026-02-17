from __future__ import annotations

"""Shared scikit-learn typing aliases used across the Engine internals.

The Engine boundary contracts are strongly typed (Pydantic models). Internally we
also want consistent, lightweight typing without importing sklearn symbols in
dozens of places.

These aliases intentionally mirror the scikit-learn mixin classes:
- BaseEstimator: common estimator interface
- ClassifierMixin / RegressorMixin / ClusterMixin: estimator-family capabilities
"""

from typing import TypeAlias

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, ClusterMixin
from sklearn.pipeline import Pipeline


SkEstimator: TypeAlias = BaseEstimator
SkClassifier: TypeAlias = ClassifierMixin
SkRegressor: TypeAlias = RegressorMixin
SkClusterer: TypeAlias = ClusterMixin

SkPipeline: TypeAlias = Pipeline
SkModel: TypeAlias = BaseEstimator | Pipeline
