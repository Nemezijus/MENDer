"""XGBoost ensemble accumulator (regression).

XGBoost reporting is shared across tasks; this module exists to keep a consistent
file layout with other ensemble families.
"""

from .accumulate_common import XGBoostEnsembleReportAccumulator

__all__ = ["XGBoostEnsembleReportAccumulator"]
