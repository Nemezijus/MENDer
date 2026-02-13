from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from engine.components.prediction.decoder.api import build_decoder_outputs_from_arrays
from engine.reporting.common.json_safety import dedupe_preserve_order
from engine.reporting.training.regression_payloads import build_regression_decoder_outputs_payload

from .types import DecoderState


def init_decoder_state(cfg, *, eval_kind: str) -> Tuple[DecoderState, Any]:
    decoder_cfg = getattr(getattr(cfg, "eval", None), "decoder", None)
    enabled = bool(getattr(decoder_cfg, "enabled", False)) if decoder_cfg is not None else False
    max_preview_rows = int(getattr(decoder_cfg, "max_preview_rows", 100) or 100) if decoder_cfg is not None else 100

    state = DecoderState(
        enabled=enabled,
        max_preview_rows=max_preview_rows,
        positive_label=getattr(decoder_cfg, "positive_class_label", None) if decoder_cfg is not None else None,
        include_scores=bool(getattr(decoder_cfg, "include_decision_scores", True)) if decoder_cfg is not None else True,
        include_probabilities=bool(getattr(decoder_cfg, "include_probabilities", True)) if decoder_cfg is not None else True,
        calibrate_probabilities=bool(getattr(decoder_cfg, "calibrate_probabilities", False)) if decoder_cfg is not None else False,
        include_margin=bool(getattr(decoder_cfg, "include_margin", False)) if decoder_cfg is not None else False,
    )

    # Only classification supports decision scores / proba decoder outputs.
    if eval_kind != "classification":
        state.enabled = enabled

    return state, decoder_cfg


def build_decoder_payload(
    *,
    decoder_state: DecoderState,
    decoder_cfg,
    eval_kind: str,
    mode: str,
    y_true_all: Optional[np.ndarray],
    y_pred_all: np.ndarray,
    row_indices: np.ndarray,
    order: Optional[np.ndarray],
    fold_ids_all: Optional[np.ndarray],
) -> Optional[Any]:
    if not decoder_state.enabled:
        return None

    # Classification: aggregated OOF scores/proba/margins
    if eval_kind == "classification":
        try:
            ds_arr = np.concatenate(decoder_state.scores_all, axis=0) if decoder_state.scores_all else None
            pr_arr = np.concatenate(decoder_state.proba_all, axis=0) if decoder_state.proba_all else None
            mg_arr = np.concatenate(decoder_state.margin_all, axis=0) if decoder_state.margin_all else None
            fold_ids_arr = np.concatenate(decoder_state.fold_ids_parts, axis=0) if decoder_state.fold_ids_parts else None

            if order is not None:
                if ds_arr is not None and ds_arr.shape[0] == order.shape[0]:
                    ds_arr = ds_arr[order]
                if pr_arr is not None and pr_arr.shape[0] == order.shape[0]:
                    pr_arr = pr_arr[order]
                if mg_arr is not None and mg_arr.shape[0] == order.shape[0]:
                    mg_arr = mg_arr[order]
                if fold_ids_arr is not None and fold_ids_arr.shape[0] == order.shape[0]:
                    fold_ids_arr = fold_ids_arr[order]

            return build_decoder_outputs_from_arrays(
                y_pred=y_pred_all,
                y_true=(y_true_all if (y_true_all is not None and np.asarray(y_true_all).size) else None),
                indices=row_indices,
                classes=decoder_state.classes,
                positive_class_label=decoder_state.positive_label,
                positive_class_index=decoder_state.positive_index,
                decision_scores=ds_arr,
                proba=pr_arr,
                proba_source=decoder_state.proba_source,
                margin=mg_arr,
                fold_ids=fold_ids_arr,
                notes=dedupe_preserve_order(decoder_state.notes),
                max_preview_rows=(decoder_state.max_preview_rows if decoder_state.max_preview_rows > 0 else None),
                include_summary=True,
            )
        except Exception:
            return None

    # Regression: summary-only payload
    if eval_kind == "regression":
        try:
            return build_regression_decoder_outputs_payload(
                decoder_cfg=decoder_cfg,
                mode=mode,
                y_true_all=y_true_all if (y_true_all is not None and np.asarray(y_true_all).size) else None,
                y_pred_all=y_pred_all,
                row_indices=row_indices,
                fold_ids_all=fold_ids_all,
                notes=[],
            )
        except Exception:
            return None

    return None
