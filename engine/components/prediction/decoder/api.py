from __future__ import annotations

"""Canonical prediction/decoder API.

This module turns raw per-sample decoder arrays (scores/probabilities/margins)
into the result contract used by the rest of MENDer:
``engine.contracts.results.decoder.DecoderOutputs``.
"""

from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

from engine.core.shapes import coerce_1d, coerce_2d_optional

from engine.contracts.results.decoder import DecoderOutputs

from ..decoder_extraction import RawDecoderOutputs, compute_decoder_outputs_raw

from engine.reporting.prediction.prediction_results import build_decoder_output_table
from engine.reporting.decoder.decoder_summaries import compute_decoder_summaries

from .errors import DecoderOutputsConcatError
from .extraction import compute_margin
from .formatting import append_concise_summary_notes, build_decoder_outputs_payload
from .inputs import ensure_indices


def predict_decoder_outputs(
    model: Any,
    X: Any,
    *,
    y_true: Optional[Any] = None,
    indices: Optional[Iterable[int]] = None,
    fold_ids: Optional[Iterable[int]] = None,
    positive_class_label: Optional[Any] = None,
    include_decision_scores: bool = True,
    include_probabilities: bool = True,
    include_margin: bool = True,
    calibrate_probabilities: bool = False,
    max_preview_rows: Optional[int] = 200,
    include_summary: bool = True,
    allow_vote_share_losses: bool = False,
) -> DecoderOutputs:
    """Compute decoder outputs using the canonical engine path."""

    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D; got {X_arr.shape}.")

    raw: RawDecoderOutputs = compute_decoder_outputs_raw(
        model,
        X_arr,
        positive_class_label=positive_class_label,
        include_decision_scores=include_decision_scores,
        include_probabilities=include_probabilities,
        calibrate_probabilities=calibrate_probabilities,
        include_margin=include_margin,
    )

    n = int(X_arr.shape[0])
    idx_list = ensure_indices(indices, n)

    fold_ids_list: Optional[List[int]] = None
    if fold_ids is not None:
        try:
            fold_ids_list = [int(v) for v in list(fold_ids)]
        except Exception:
            fold_ids_list = None
        if fold_ids_list is not None and len(fold_ids_list) != n:
            fold_ids_list = None

    # Use full arrays for summary; table builder will slice to preview.
    ds = np.asarray(raw.decision_scores) if (raw.decision_scores is not None and include_decision_scores) else None
    pr = np.asarray(raw.proba) if (raw.proba is not None and include_probabilities) else None
    mg = np.asarray(raw.margin) if (raw.margin is not None and include_margin) else None

    rows = build_decoder_output_table(
        indices=idx_list,
        fold_ids=fold_ids_list,
        y_pred=coerce_1d(raw.y_pred),
        y_true=coerce_1d(y_true) if y_true is not None else None,
        classes=np.asarray(raw.classes) if raw.classes is not None else None,
        decision_scores=ds,
        proba=pr,
        margin=mg,
        positive_class_label=raw.positive_class_label,
        positive_class_index=raw.positive_class_index,
        max_rows=max_preview_rows,
    )

    notes: List[str] = []
    notes.extend([str(n) for n in (raw.notes or [])])

    summary = None
    if include_summary:
        summary, summary_notes = compute_decoder_summaries(
            y_true=y_true,
            classes=np.asarray(raw.classes) if raw.classes is not None else None,
            proba=pr,
            proba_source=raw.proba_source,
            decision_scores=ds,
            margin=mg,
            allow_vote_share_losses=allow_vote_share_losses,
        )
        notes.extend([str(n) for n in (summary_notes or [])])

    payload = build_decoder_outputs_payload(
        classes=raw.classes,
        positive_class_label=raw.positive_class_label,
        positive_class_index=raw.positive_class_index,
        has_decision_scores=ds is not None,
        has_proba=pr is not None,
        proba_source=raw.proba_source,
        notes=notes,
        preview_rows=rows,
        n_rows_total=n,
        summary=summary,
    )
    return DecoderOutputs.model_validate(payload)


def build_decoder_outputs_from_arrays(
    *,
    y_pred: Any,
    y_true: Optional[Any],
    indices: Optional[Iterable[int]],
    classes: Optional[Any],
    positive_class_label: Optional[Any],
    positive_class_index: Optional[int],
    decision_scores: Optional[Any],
    proba: Optional[Any],
    proba_source: Optional[str],
    margin: Optional[Any],
    fold_ids: Optional[Iterable[int]],
    notes: Sequence[str] = (),
    max_preview_rows: Optional[int] = 200,
    include_summary: bool = True,
    allow_vote_share_losses: bool = False,
) -> DecoderOutputs:
    """Build DecoderOutputs contract from already-computed arrays."""

    y_pred_arr = coerce_1d(y_pred)
    n = int(y_pred_arr.shape[0])

    idx_list = ensure_indices(indices, n)

    fold_ids_list: Optional[List[int]] = None
    if fold_ids is not None:
        try:
            fold_ids_list = [int(v) for v in list(fold_ids)]
        except Exception:
            fold_ids_list = None
        if fold_ids_list is not None and len(fold_ids_list) != n:
            fold_ids_list = None

    classes_arr = np.asarray(classes) if classes is not None else None

    # Determine positive class index if needed
    pos_idx = positive_class_index
    if pos_idx is None and positive_class_label is not None and classes_arr is not None:
        try:
            matches = np.where(classes_arr == positive_class_label)[0]
            if matches.size > 0:
                pos_idx = int(matches[0])
        except Exception:
            pass

    ds_arr = np.asarray(decision_scores) if decision_scores is not None else None
    pr_arr = coerce_2d_optional(proba)
    mg_arr = np.asarray(margin) if margin is not None else None
    if mg_arr is None:
        mg_arr = compute_margin(decision_scores=ds_arr, proba=pr_arr)

    rows = build_decoder_output_table(
        indices=idx_list,
        fold_ids=fold_ids_list,
        y_pred=y_pred_arr,
        y_true=coerce_1d(y_true) if y_true is not None else None,
        classes=classes_arr,
        decision_scores=ds_arr,
        proba=pr_arr,
        margin=mg_arr,
        positive_class_label=positive_class_label,
        positive_class_index=pos_idx,
        max_rows=max_preview_rows,
    )

    all_notes: List[str] = [str(x) for x in notes]

    summary = None
    if include_summary:
        summary, summary_notes = compute_decoder_summaries(
            y_true=y_true,
            classes=classes_arr,
            proba=pr_arr,
            proba_source=proba_source,
            decision_scores=ds_arr,
            margin=mg_arr,
            allow_vote_share_losses=allow_vote_share_losses,
        )
        all_notes.extend([str(n) for n in (summary_notes or [])])

    payload = build_decoder_outputs_payload(
        classes=classes_arr,
        positive_class_label=positive_class_label,
        positive_class_index=pos_idx,
        has_decision_scores=ds_arr is not None,
        has_proba=pr_arr is not None,
        proba_source=proba_source,
        notes=all_notes,
        preview_rows=rows,
        n_rows_total=n,
        summary=summary,
    )
    return DecoderOutputs.model_validate(payload)


def build_decoder_outputs_from_parts(
    *,
    decoder_cfg: Any,
    mode: str,
    y_pred_all: np.ndarray,
    y_true_all: Optional[np.ndarray],
    order: Optional[np.ndarray],
    row_indices: np.ndarray,
    decoder_classes: Optional[np.ndarray],
    positive_class_label: Any,
    positive_class_index: Optional[int],
    decision_scores_parts: List[Optional[np.ndarray]],
    proba_parts: List[Optional[np.ndarray]],
    margin_parts: List[Optional[np.ndarray]],
    fold_ids_parts: List[np.ndarray],
    notes: List[str],
) -> DecoderOutputs:
    """Train-service helper: build decoder outputs from fold parts."""

    def _concat_if_complete(parts: List[Optional[np.ndarray]], name: str) -> Optional[np.ndarray]:
        if not parts:
            return None
        if any(p is None for p in parts):
            notes.append(f"Decoder outputs: '{name}' omitted because at least one fold could not produce it.")
            return None
        try:
            return np.concatenate([np.asarray(p) for p in parts], axis=0)
        except Exception as e:
            raise DecoderOutputsConcatError(
                f"Cannot concatenate decoder outputs '{name}' ({type(e).__name__}: {e})"
            ) from e

    ds_arr = _concat_if_complete(decision_scores_parts, "decision_scores")
    pr_arr = _concat_if_complete(proba_parts, "proba")
    mg_arr = _concat_if_complete(margin_parts, "margin")
    fold_ids_arr = np.concatenate(fold_ids_parts, axis=0) if fold_ids_parts else None

    # Reorder to original row order if we have a stable mapping
    if order is not None:
        if ds_arr is not None and ds_arr.shape[0] == order.shape[0]:
            ds_arr = ds_arr[order]
        if pr_arr is not None and pr_arr.shape[0] == order.shape[0]:
            pr_arr = pr_arr[order]
        if mg_arr is not None and mg_arr.shape[0] == order.shape[0]:
            mg_arr = mg_arr[order]
        if fold_ids_arr is not None and fold_ids_arr.shape[0] == order.shape[0]:
            fold_ids_arr = fold_ids_arr[order]

    # For holdout, set fold_id=1 to match prior UI expectations.
    if fold_ids_arr is None and str(mode).lower() == "holdout":
        try:
            fold_ids_arr = np.ones((int(np.asarray(y_pred_all).shape[0]),), dtype=int)
        except Exception:
            fold_ids_arr = None

    max_rows = int(getattr(decoder_cfg, "max_preview_rows", 200)) if decoder_cfg is not None else 200

    indices_out = [int(v) for v in (row_indices.tolist() if row_indices is not None else [])]

    dec = build_decoder_outputs_from_arrays(
        y_pred=y_pred_all,
        y_true=y_true_all if y_true_all is not None and np.asarray(y_true_all).size else None,
        indices=indices_out,
        classes=decoder_classes,
        positive_class_label=positive_class_label,
        positive_class_index=positive_class_index,
        decision_scores=ds_arr,
        proba=pr_arr,
        proba_source=None,
        margin=mg_arr,
        fold_ids=fold_ids_arr,
        notes=notes,
        max_preview_rows=max_rows,
        include_summary=True,
    )

    append_concise_summary_notes(notes, dec.summary)
    return dec
