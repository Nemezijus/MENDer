from __future__ import annotations

from typing import Any, Optional

import numpy as np

from engine.reporting.training.decoder_payloads import build_decoder_outputs_payload
from engine.reporting.training.regression_payloads import build_regression_decoder_outputs_payload

from .types import DecoderParts


def build_decoder_outputs(
    *,
    decoder_cfg,
    mode: str,
    eval_kind: str,
    y_true_all: Optional[np.ndarray],
    y_pred_all: np.ndarray,
    row_indices: np.ndarray,
    order: Optional[np.ndarray],
    fold_ids_all: Optional[np.ndarray],
    decoder_parts: DecoderParts,
    positive_class_label,
) -> Optional[Any]:
    """Build a decoder_outputs payload for TrainResult.

    Returns
    -------
    Decoder outputs payload object (pydantic model or dict-like), or None.
    """

    decoder_enabled = bool(getattr(decoder_cfg, "enabled", False)) if decoder_cfg is not None else False
    if not decoder_enabled:
        return None

    if eval_kind == "classification":
        try:
            return build_decoder_outputs_payload(
                decoder_cfg=decoder_cfg,
                mode=mode,
                y_pred_all=y_pred_all,
                y_true_all=y_true_all if (y_true_all is not None and y_true_all.size) else None,
                order=order,
                row_indices=row_indices,
                decoder_classes=decoder_parts.classes,
                positive_class_label=positive_class_label,
                positive_class_index=decoder_parts.positive_index,
                decision_scores_parts=decoder_parts.scores_parts,
                proba_parts=decoder_parts.proba_parts,
                margin_parts=decoder_parts.margin_parts,
                fold_ids_parts=decoder_parts.fold_ids_parts,
                notes=decoder_parts.notes,
            )
        except Exception:
            return None

    if eval_kind == "regression":
        try:
            return build_regression_decoder_outputs_payload(
                decoder_cfg=decoder_cfg,
                mode=mode,
                y_true_all=y_true_all if (y_true_all is not None and y_true_all.size) else None,
                y_pred_all=y_pred_all,
                row_indices=row_indices,
                fold_ids_all=fold_ids_all,
                notes=[],
            )
        except Exception:
            return None

    return None
