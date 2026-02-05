from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine.contracts.results.decoder import DecoderOutputs

from engine.components.prediction.decoder_api import build_decoder_outputs_from_parts

def build_decoder_outputs_payload(
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
    """Assemble the decoder_outputs payload for TrainResponse.

    Segment 6 canonicalization: delegate assembly to the engine-level
    decoder API so there is only one way to build decoder outputs.
    """

    return build_decoder_outputs_from_parts(
        decoder_cfg=decoder_cfg,
        mode=mode,
        y_pred_all=y_pred_all,
        y_true_all=y_true_all,
        order=order,
        row_indices=row_indices,
        decoder_classes=decoder_classes,
        positive_class_label=positive_class_label,
        positive_class_index=positive_class_index,
        decision_scores_parts=decision_scores_parts,
        proba_parts=proba_parts,
        margin_parts=margin_parts,
        fold_ids_parts=fold_ids_parts,
        notes=notes,
    )
