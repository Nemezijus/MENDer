from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from engine.contracts.results.decoder import DecoderOutputs

from engine.reporting.decoder.decoder_outputs import compute_decoder_outputs
from engine.reporting.prediction.prediction_results import (
    build_decoder_output_table,
    merge_prediction_and_decoder_tables,
)


def add_decoder_outputs_preview(
    *,
    result: Dict[str, Any],
    pipeline: Any,
    X_arr: np.ndarray,
    y_arr: Optional[np.ndarray],
    ev: Any,
    eval_kind: str,
    max_preview_rows: int,
) -> None:
    """Mutates result dict by adding 'decoder_outputs' (classification-only)."""
    decoder_cfg = getattr(ev, "decoder", None)
    decoder_enabled = bool(getattr(decoder_cfg, "enabled", False)) if decoder_cfg is not None else False
    if not (decoder_enabled and eval_kind == "classification"):
        return

    decoder_positive_label = getattr(decoder_cfg, "positive_class_label", None) if decoder_cfg is not None else None
    decoder_include_scores = bool(getattr(decoder_cfg, "include_decision_scores", True)) if decoder_cfg is not None else True
    decoder_include_probabilities = bool(getattr(decoder_cfg, "include_probabilities", True)) if decoder_cfg is not None else True
    decoder_include_margin = bool(getattr(decoder_cfg, "include_margin", True)) if decoder_cfg is not None else True
    decoder_calibrate_probabilities = bool(getattr(decoder_cfg, "calibrate_probabilities", False)) if decoder_cfg is not None else False

    n_samples = int(X_arr.shape[0])
    decoder_preview_cap = (
        int(getattr(decoder_cfg, "max_preview_rows", max_preview_rows) or max_preview_rows)
        if decoder_cfg is not None
        else max_preview_rows
    )
    n_preview_dec = min(decoder_preview_cap, n_samples)

    try:
        dec = compute_decoder_outputs(
            pipeline,
            X_arr,
            positive_class_label=decoder_positive_label,
            include_decision_scores=decoder_include_scores,
            include_probabilities=decoder_include_probabilities,
            calibrate_probabilities=decoder_calibrate_probabilities,
        )

        ds_preview = (
            np.asarray(dec.decision_scores)[:n_preview_dec]
            if (dec.decision_scores is not None and decoder_include_scores)
            else None
        )
        pr_preview = (
            np.asarray(dec.proba)[:n_preview_dec]
            if (dec.proba is not None and decoder_include_probabilities)
            else None
        )
        mg_preview = (
            np.asarray(dec.margin)[:n_preview_dec]
            if (decoder_include_margin and dec.margin is not None)
            else None
        )

        y_true_dec_preview = y_arr[:n_preview_dec] if y_arr is not None else None
        y_pred_dec_preview = np.asarray(dec.y_pred).ravel()[:n_preview_dec]

        rows = build_decoder_output_table(
            indices=list(range(n_preview_dec)),
            y_pred=y_pred_dec_preview,
            y_true=y_true_dec_preview,
            classes=dec.classes,
            decision_scores=ds_preview,
            proba=pr_preview,
            margin=mg_preview,
            positive_class_label=dec.positive_class_label,
            positive_class_index=dec.positive_class_index,
        )

        payload = {
            "classes": (dec.classes.tolist() if hasattr(dec.classes, "tolist") else dec.classes)
            if dec.classes is not None
            else None,
            "positive_class_label": dec.positive_class_label,
            "positive_class_index": dec.positive_class_index,
            "has_decision_scores": bool(dec.decision_scores is not None and decoder_include_scores),
            "has_proba": bool(dec.proba is not None and decoder_include_probabilities),
            "notes": list(dec.notes or []),
            "preview_rows": rows,
            "n_rows_total": int(n_samples),
        }
        result["decoder_outputs"] = DecoderOutputs.model_validate(payload)
    except Exception as e:
        payload = {
            "classes": None,
            "positive_class_label": decoder_positive_label,
            "positive_class_index": None,
            "has_decision_scores": False,
            "has_proba": False,
            "notes": [f"Decoder outputs could not be computed ({type(e).__name__}: {e})"],
            "preview_rows": [],
            "n_rows_total": None,
        }
        result["decoder_outputs"] = DecoderOutputs.model_validate(payload)


def maybe_merge_decoder_into_export_table(
    *,
    table: Any,
    pipeline: Any,
    X_arr: np.ndarray,
    y_arr: Optional[np.ndarray],
    ev: Any,
    eval_kind: str,
) -> Any:
    """For export flow: add decoder columns into full prediction table when enabled."""
    decoder_cfg = getattr(ev, "decoder", None)
    decoder_enabled = bool(getattr(decoder_cfg, "enabled", False)) if decoder_cfg is not None else False
    decoder_export_enabled = bool(getattr(decoder_cfg, "enable_export", True)) if decoder_cfg is not None else True

    if not (decoder_enabled and decoder_export_enabled and eval_kind == "classification"):
        return table

    decoder_positive_label = getattr(decoder_cfg, "positive_class_label", None) if decoder_cfg is not None else None
    decoder_include_scores = bool(getattr(decoder_cfg, "include_decision_scores", True)) if decoder_cfg is not None else True
    decoder_include_probabilities = bool(getattr(decoder_cfg, "include_probabilities", True)) if decoder_cfg is not None else True
    decoder_include_margin = bool(getattr(decoder_cfg, "include_margin", True)) if decoder_cfg is not None else True
    decoder_calibrate_probabilities = bool(getattr(decoder_cfg, "calibrate_probabilities", False)) if decoder_cfg is not None else False

    n_samples = int(X_arr.shape[0])

    try:
        dec = compute_decoder_outputs(
            pipeline,
            X_arr,
            positive_class_label=decoder_positive_label,
            include_decision_scores=decoder_include_scores,
            include_probabilities=decoder_include_probabilities,
            calibrate_probabilities=decoder_calibrate_probabilities,
        )
        ds = dec.decision_scores if (dec.decision_scores is not None and decoder_include_scores) else None
        pr = dec.proba if (dec.proba is not None and decoder_include_probabilities) else None
        mg = dec.margin if (dec.margin is not None and decoder_include_margin) else None
        y_pred_dec = np.asarray(dec.y_pred).ravel()

        decoder_rows = build_decoder_output_table(
            indices=range(n_samples),
            y_pred=y_pred_dec,
            y_true=y_arr,
            classes=dec.classes,
            decision_scores=ds,
            proba=pr,
            margin=mg,
            positive_class_label=dec.positive_class_label,
            positive_class_index=dec.positive_class_index,
            max_rows=None,
        )
        return merge_prediction_and_decoder_tables(
            prediction_rows=table,
            decoder_rows=decoder_rows,
        )
    except Exception:
        return table
