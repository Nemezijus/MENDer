"""Shared accumulator building blocks for ensemble reporting.

This package exists to reduce drift across ensemble-family accumulators (voting/bagging/
AdaBoost/XGBoost) and across task types (classification/regression).
"""

from .base import FoldAccumulatorBase

__all__ = [
    "FoldAccumulatorBase",
]
