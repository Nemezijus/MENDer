"""Regression diagnostics helpers.

This is an internal subpackage; the stable public entrypoint is
:mod:`engine.reporting.diagnostics.regression_diagnostics`.
"""

from .summary_metrics import regression_summary
from .histograms import histogram_1d
from .residuals import downsample_xy, binned_error_by_true

__all__ = [
    "regression_summary",
    "histogram_1d",
    "downsample_xy",
    "binned_error_by_true",
]
