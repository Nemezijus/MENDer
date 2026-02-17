"""XGBoost ensemble builder entrypoints.

This package exposes the public API:

  - build_xgboost_ensemble
  - fit_xgboost_ensemble
  - XGBClassifierLabelAdapter

Implementation details live in dedicated modules to keep SRP boundaries clear.
"""

from __future__ import annotations

from .build import build_xgboost_ensemble
from .fit import fit_xgboost_ensemble
from .labels import XGBClassifierLabelAdapter

__all__ = ["build_xgboost_ensemble", "fit_xgboost_ensemble", "XGBClassifierLabelAdapter"]
