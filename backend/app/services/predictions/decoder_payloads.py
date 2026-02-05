from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from engine.contracts.results.decoder import DecoderOutputs

from engine.components.prediction import predict_decoder_outputs
from engine.reporting.prediction.prediction_results import merge_prediction_and_decoder_tables


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
        dec = predict_decoder_outputs(
            pipeline,
            X_arr,
            y_true=y_arr,
            indices=range(n_samples),
            positive_class_label=decoder_positive_label,
            include_decision_scores=decoder_include_scores,
            include_probabilities=decoder_include_probabilities,
            include_margin=decoder_include_margin,
            calibrate_probabilities=decoder_calibrate_probabilities,
            max_preview_rows=n_preview_dec,
            include_summary=False,
        )
        result["decoder_outputs"] = dec
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
        dec = predict_decoder_outputs(
            pipeline,
            X_arr,
            y_true=y_arr,
            indices=range(n_samples),
            positive_class_label=decoder_positive_label,
            include_decision_scores=decoder_include_scores,
            include_probabilities=decoder_include_probabilities,
            include_margin=decoder_include_margin,
            calibrate_probabilities=decoder_calibrate_probabilities,
            max_preview_rows=None,
            include_summary=False,
        )

        decoder_rows = [r.model_dump() for r in (dec.preview_rows or [])]
        return merge_prediction_and_decoder_tables(
            prediction_rows=table,
            decoder_rows=decoder_rows,
        )
    except Exception:
        return table
