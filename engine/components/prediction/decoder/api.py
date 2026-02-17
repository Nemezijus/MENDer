from __future__ import annotations

"""Canonical prediction/decoder API.

This module exposes the public decoder-output entrypoints used throughout MENDer.

SRP note:
- raw extraction from estimators lives in ``engine.components.prediction.decoder_extraction``
- contract assembly (tables + summaries + payload validation) lives in ``.assembly``
- fold concatenation utilities live in ``.concat``

Keeping this file as a *thin orchestrator* makes it easier to test each concern
in isolation while preserving the stable import path:
``engine.components.prediction.decoder.api``.
"""

from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

from engine.contracts.results.decoder import DecoderOutputs
from engine.core.shapes import coerce_1d

from ..decoder_extraction import RawDecoderOutputs, compute_decoder_outputs_raw

from .assembly import assemble_decoder_outputs
from .concat import concat_if_complete, reorder_if_possible
from .folds import ensure_holdout_fold_ids, parse_fold_ids
from .formatting import append_concise_summary_notes
from .inputs import ensure_indices
from .positive import resolve_positive_class_index


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
    fold_ids_list = parse_fold_ids(fold_ids, n)

    ds = raw.decision_scores if (raw.decision_scores is not None and include_decision_scores) else None
    pr = raw.proba if (raw.proba is not None and include_probabilities) else None
    mg = raw.margin if (raw.margin is not None and include_margin) else None

    notes: List[str] = [str(n) for n in (raw.notes or [])]

    return assemble_decoder_outputs(
        indices=idx_list,
        fold_ids=fold_ids_list,
        y_pred=raw.y_pred,
        y_true=y_true,
        classes=np.asarray(raw.classes) if raw.classes is not None else None,
        decision_scores=ds,
        proba=pr,
        proba_source=raw.proba_source,
        margin=mg,
        positive_class_label=raw.positive_class_label,
        positive_class_index=raw.positive_class_index,
        notes=notes,
        max_preview_rows=max_preview_rows,
        include_summary=include_summary,
        allow_vote_share_losses=allow_vote_share_losses,
    )


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
    fold_ids_list = parse_fold_ids(fold_ids, n)

    classes_arr = np.asarray(classes) if classes is not None else None
    pos_idx = resolve_positive_class_index(
        classes=classes_arr,
        positive_class_label=positive_class_label,
        positive_class_index=positive_class_index,
    )

    all_notes: List[str] = [str(x) for x in notes]

    return assemble_decoder_outputs(
        indices=idx_list,
        fold_ids=fold_ids_list,
        y_pred=y_pred,
        y_true=y_true,
        classes=classes_arr,
        decision_scores=decision_scores,
        proba=proba,
        proba_source=proba_source,
        margin=margin,
        positive_class_label=positive_class_label,
        positive_class_index=pos_idx,
        notes=all_notes,
        max_preview_rows=max_preview_rows,
        include_summary=include_summary,
        allow_vote_share_losses=allow_vote_share_losses,
    )


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

    ds_arr = concat_if_complete(decision_scores_parts, name="decision_scores", notes=notes)
    pr_arr = concat_if_complete(proba_parts, name="proba", notes=notes)
    mg_arr = concat_if_complete(margin_parts, name="margin", notes=notes)
    fold_ids_arr = np.concatenate(fold_ids_parts, axis=0) if fold_ids_parts else None

    # Reorder to original row order if we have a stable mapping
    ds_arr = reorder_if_possible(ds_arr, order)
    pr_arr = reorder_if_possible(pr_arr, order)
    mg_arr = reorder_if_possible(mg_arr, order)
    fold_ids_arr = reorder_if_possible(fold_ids_arr, order)

    # For holdout, set fold_id=1 to match prior UI expectations.
    n = int(np.asarray(y_pred_all).shape[0])
    fold_ids_arr = ensure_holdout_fold_ids(mode=mode, fold_ids=fold_ids_arr, n=n)

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

    # Mutate the outer `notes` list (used elsewhere) with a concise summary.
    append_concise_summary_notes(notes, dec.summary)
    return dec
