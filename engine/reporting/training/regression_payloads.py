from __future__ import annotations

"""Regression reporting helpers used at training time."""

from typing import Any, Dict, List, Optional

import numpy as np

from engine.contracts.results.decoder import DecoderOutputs
from engine.reporting.common.json_safety import ReportError
from engine.reporting.common.report_errors import record_error
from engine.reporting.diagnostics.regression_diagnostics import (
    regression_diagnostics,
    regression_summary,
)
from engine.reporting.prediction.prediction_results import build_prediction_table


def build_regression_diagnostics_payload(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seed: int = 0,
    max_points: int = 5000,
    n_hist_bins: int = 30,
    n_bins_by_true: int = 10,
) -> Dict[str, Any]:
    """Compute regression diagnostics and return JSON-friendly payload."""

    payload: Dict[str, Any] = {}

    try:
        diag = regression_diagnostics(
            y_true=y_true,
            y_pred=y_pred,
            scatter_max_points=int(max_points),
            residual_hist_bins=int(n_hist_bins),
            seed=int(seed),
        )
        payload = dict(diag or {})
    except Exception as e:
        record_error(payload, where="reporting.training.regression_diagnostics", exc=e)
        return payload

    # Normalize summary keys to the API schema
    try:
        summ = payload.get("summary")
        if isinstance(summ, dict):
            if "median_abs_error" in summ and "median_ae" not in summ:
                summ["median_ae"] = summ.get("median_abs_error")
    except Exception as e:
        record_error(payload, where="reporting.training.regression_payloads.summary_normalization", exc=e)

    return payload


def build_regression_decoder_outputs_payload(
    *,
    decoder_cfg: Any,
    mode: str,
    y_true_all: Optional[np.ndarray],
    y_pred_all: np.ndarray,
    row_indices: np.ndarray,
    fold_ids_all: Optional[np.ndarray],
    notes: List[str],
) -> DecoderOutputs:
    """Build a DecoderOutputs-like contract for regression.

    For regression we do not have decision scores / probabilities.
    We still provide a per-sample preview table and a compact summary.

    Note: DecoderOutputs is a strict contract (extra fields forbidden). We surface
    any reporting issues as additional note strings.
    """

    max_rows = (
        int(getattr(decoder_cfg, "max_preview_rows", 200) or 200) if decoder_cfg is not None else 200
    )

    indices_out = [int(v) for v in (row_indices.tolist() if row_indices is not None else [])]
    y_true_use = y_true_all if y_true_all is not None and np.asarray(y_true_all).size else None

    rows = build_prediction_table(
        indices=indices_out,
        y_pred=y_pred_all,
        y_true=y_true_use,
        task="regression",
        max_rows=max_rows,
    )

    # Add fold_id to preview rows when available
    if fold_ids_all is not None and len(rows) > 0:
        for i, r in enumerate(rows):
            try:
                r["fold_id"] = int(np.asarray(fold_ids_all).ravel()[i])
            except Exception as e:
                notes.append(f"Decoder preview: failed to attach fold_id for row {i}: {type(e).__name__}: {e}")
    elif mode == "holdout" and len(rows) > 0:
        for r in rows:
            r["fold_id"] = 1

    summary = None
    try:
        if y_true_use is not None:
            summary = regression_summary(y_true=y_true_use, y_pred=y_pred_all)
            # normalize key
            if isinstance(summary, dict) and "median_abs_error" in summary and "median_ae" not in summary:
                summary["median_ae"] = summary.get("median_abs_error")
    except Exception as e:
        summary = None
        notes.append(f"Decoder summary failed: {type(e).__name__}: {e}")

    payload = {
        "classes": None,
        "positive_class_label": None,
        "positive_class_index": None,
        "has_decision_scores": False,
        "has_proba": False,
        "proba_source": None,
        "summary": summary,
        "notes": notes,
        "n_rows_total": int(np.asarray(y_pred_all).shape[0]) if y_pred_all is not None else 0,
        "preview_rows": rows,
    }

    return DecoderOutputs.model_validate(payload)
