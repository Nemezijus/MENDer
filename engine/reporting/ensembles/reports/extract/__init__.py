"""Shared extraction utilities used by ensemble report updaters.

These helpers intentionally live under `reports/` (not the accumulators) because
their job is to *extract* and *normalize* data from trained estimators/pipelines
for reporting, without owning any aggregation state.
"""

from .pipeline import resolve_estimator_and_X
from .base_predictions import (
    collect_base_predictions_classification,
    collect_base_predictions_regression,
)
from .voting import collect_voting_base_preds_and_scores
from .xgboost import extract_xgboost_fold_info

__all__ = [
    "resolve_estimator_and_X",
    "collect_base_predictions_classification",
    "collect_base_predictions_regression",
    "collect_voting_base_preds_and_scores",
    "extract_xgboost_fold_info",
]
