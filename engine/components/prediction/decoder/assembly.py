from __future__ import annotations

"""Assemble the DecoderOutputs result contract from already-computed arrays.

This module owns *formatting/assembly* responsibilities:
- build preview rows (table)
- compute summary metrics (optional)
- build the JSON payload and validate the DecoderOutputs contract

It intentionally does **not** extract raw arrays from models. That responsibility
lives in ``engine.components.prediction.decoder_extraction``.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from engine.core.shapes import coerce_1d, coerce_2d_optional
from engine.contracts.results.decoder import DecoderOutputs

from engine.reporting.prediction.prediction_results import build_decoder_output_table
from engine.reporting.decoder.decoder_summaries import compute_decoder_summaries

from .extraction import compute_margin
from .formatting import build_decoder_outputs_payload


def assemble_decoder_outputs(
    *,
    indices: List[int],
    fold_ids: Optional[List[int]],
    y_pred: Any,
    y_true: Optional[Any],
    classes: Optional[np.ndarray],
    decision_scores: Optional[Any],
    proba: Optional[Any],
    proba_source: Optional[str],
    margin: Optional[Any],
    positive_class_label: Optional[Any],
    positive_class_index: Optional[int],
    notes: List[str],
    max_preview_rows: Optional[int] = 200,
    include_summary: bool = True,
    allow_vote_share_losses: bool = False,
) -> DecoderOutputs:
    """Create a DecoderOutputs contract from arrays.

    Args:
        indices: Row indices (length n).
        fold_ids: Fold id per row (length n) or None.
        y_pred: Predicted label per row.
        y_true: Ground truth label per row (optional).
        classes: Class labels (optional; required for some summaries).
        decision_scores: Decision scores (optional, 1D or 2D).
        proba: Probabilities or vote shares (optional, 2D).
        proba_source: Informational tag about where probabilities came from.
        margin: Confidence proxy (optional). If None, computed best-effort.
        positive_class_label/index: Optional definition of the positive class.
        notes: Mutable list of notes (strings).
    """

    y_pred_arr = coerce_1d(y_pred)
    n = int(y_pred_arr.shape[0])

    if len(indices) != n:
        raise ValueError(f"indices length must equal n={n}; got {len(indices)}")
    if fold_ids is not None and len(fold_ids) != n:
        fold_ids = None

    y_true_arr = coerce_1d(y_true) if y_true is not None else None

    ds_arr = np.asarray(decision_scores) if decision_scores is not None else None
    pr_arr = coerce_2d_optional(proba)

    mg_arr = np.asarray(margin) if margin is not None else None
    if mg_arr is None:
        mg_arr = compute_margin(decision_scores=ds_arr, proba=pr_arr)

    rows: List[Dict[str, Any]] = build_decoder_output_table(
        indices=indices,
        fold_ids=fold_ids,
        y_pred=y_pred_arr,
        y_true=y_true_arr,
        classes=classes,
        decision_scores=ds_arr,
        proba=pr_arr,
        margin=mg_arr,
        positive_class_label=positive_class_label,
        positive_class_index=positive_class_index,
        max_rows=max_preview_rows,
    )

    summary = None
    if include_summary:
        summary, summary_notes = compute_decoder_summaries(
            y_true=y_true,
            classes=classes,
            proba=pr_arr,
            proba_source=proba_source,
            decision_scores=ds_arr,
            margin=mg_arr,
            allow_vote_share_losses=allow_vote_share_losses,
        )
        if summary_notes:
            notes.extend([str(n) for n in summary_notes])

    payload = build_decoder_outputs_payload(
        classes=classes,
        positive_class_label=positive_class_label,
        positive_class_index=positive_class_index,
        has_decision_scores=ds_arr is not None,
        has_proba=pr_arr is not None,
        proba_source=proba_source,
        notes=notes,
        preview_rows=rows,
        n_rows_total=n,
        summary=summary,
    )

    return DecoderOutputs.model_validate(payload)
