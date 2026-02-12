from __future__ import annotations

"""Regression diagnostics (business-logic only).

This module is the stable public entrypoint used by the rest of the BL.
Implementation helpers live in the :mod:`engine.reporting.diagnostics.regression` package.
"""

from typing import Any, Dict, Mapping, List

from engine.reporting.common.json_safety import add_report_error, ReportError

from .regression import (
    binned_error_by_true,
    downsample_xy,
    histogram_1d,
    regression_summary,
)
from .regression.common import as_1d, numpy


def regression_diagnostics(
    *,
    y_true: Any,
    y_pred: Any,
    scatter_max_points: int = 5000,
    residual_hist_bins: int = 40,
    seed: int = 0,
) -> Mapping[str, Any]:
    """Build a full regression diagnostics payload.

    Payload includes:
      - summary metrics
      - predicted-vs-true points (downsampled)
      - residual histogram
      - residuals-vs-pred points (downsampled)
      - binned error by true quantiles
    """

    np = numpy()
    errors: List[ReportError] = []
    if np is None:
        add_report_error(errors, where="regression_diagnostics", exc=RuntimeError("numpy unavailable"))
        return {"summary": regression_summary(y_true=y_true, y_pred=y_pred), "errors": errors}

    try:
        yt = np.asarray(as_1d(y_true), dtype=float).reshape(-1)
        yp = np.asarray(as_1d(y_pred), dtype=float).reshape(-1)
    except Exception as e:
        add_report_error(errors, where="regression_diagnostics.asarray", exc=e)
        return {"summary": regression_summary(y_true=y_true, y_pred=y_pred), "errors": errors}

    if yt.size == 0 or yp.size == 0 or yt.size != yp.size:
        add_report_error(errors, where="regression_diagnostics.shape", exc=ValueError("empty/mismatched arrays"))
        return {"summary": regression_summary(y_true=y_true, y_pred=y_pred), "errors": errors}

    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]
    if yt.size == 0:
        add_report_error(errors, where="regression_diagnostics.finite", exc=ValueError("no finite samples"))
        return {"summary": regression_summary(y_true=y_true, y_pred=y_pred), "errors": errors}

    residuals = yt - yp

    payload: Dict[str, Any] = {
        "summary": regression_summary(y_true=yt, y_pred=yp),
        "pred_vs_true": downsample_xy(yt, yp, max_points=scatter_max_points, seed=seed),
        "residuals_vs_pred": downsample_xy(yp, residuals, max_points=scatter_max_points, seed=seed),
        "residual_hist": histogram_1d(residuals, n_bins=residual_hist_bins),
        "error_by_true_bin": binned_error_by_true(yt, yp, n_bins=10),
    }

    # Provide the ideal line extent for pred-vs-true (frontend can draw y=x)
    try:
        lo = float(min(float(np.min(yt)), float(np.min(yp))))
        hi = float(max(float(np.max(yt)), float(np.max(yp))))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            payload["ideal_line"] = {"x": [lo, hi], "y": [lo, hi]}
    except Exception as e:
        add_report_error(errors, where="regression_diagnostics.ideal_line", exc=e)

    if errors:
        payload["errors"] = errors

    return payload


__all__ = ["regression_diagnostics", "regression_summary"]
