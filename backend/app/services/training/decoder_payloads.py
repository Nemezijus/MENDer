from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine.contracts.results.decoder import DecoderOutputs

from engine.reporting.decoder.decoder_summaries import compute_decoder_summaries
from engine.reporting.prediction.prediction_results import build_decoder_output_table


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
) -> Dict[str, Any]:
    """Assemble the decoder_outputs payload for TrainResponse.

    For holdout: parts typically contain one entry.
    For kfold: parts contain OOF pieces per fold.
    """

    def _concat_if_complete(parts: List[Optional[np.ndarray]], name: str) -> Optional[np.ndarray]:
        if not parts:
            return None
        if any(p is None for p in parts):
            notes.append(
                f"Decoder outputs: '{name}' omitted because at least one fold could not produce it."
            )
            return None
        try:
            return np.concatenate([np.asarray(p) for p in parts], axis=0)
        except Exception as e:
            notes.append(
                f"Decoder outputs: '{name}' omitted due to concat error ({type(e).__name__}: {e})."
            )
            return None

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

    max_rows = int(getattr(decoder_cfg, "max_preview_rows", 200)) if decoder_cfg is not None else 200

    indices_out = [int(v) for v in (row_indices.tolist() if row_indices is not None else [])]

    rows = build_decoder_output_table(
        indices=indices_out,
        y_pred=y_pred_all,
        y_true=y_true_all if y_true_all is not None and np.asarray(y_true_all).size else None,
        classes=decoder_classes,
        decision_scores=ds_arr,
        proba=pr_arr,
        margin=mg_arr,
        positive_class_label=positive_class_label,
        positive_class_index=positive_class_index,
        max_rows=max_rows,
    )

    summary, summary_notes = compute_decoder_summaries(
        y_true=y_true_all if y_true_all is not None and np.asarray(y_true_all).size else None,
        classes=decoder_classes,
        proba=pr_arr,
        decision_scores=ds_arr,
        margin=mg_arr,
    )
    if summary_notes:
        notes.extend([str(n) for n in summary_notes])

    # Add concise summary notes (best-effort)
    try:
        bits = []
        if isinstance(summary, dict):
            if "log_loss" in summary:
                bits.append(f"log_loss={float(summary['log_loss']):.4f}")
            if "brier" in summary:
                bits.append(f"brier={float(summary['brier']):.4f}")
            if "margin_mean" in summary:
                bits.append(f"margin_mean={float(summary['margin_mean']):.4f}")
            if "max_proba_mean" in summary:
                bits.append(f"mean_max_proba={float(summary['max_proba_mean']):.4f}")
        if bits:
            notes.append("Decoder summary: " + ", ".join(bits))
    except Exception:
        pass

    # Add fold_id to preview rows when available
    if fold_ids_arr is not None and len(rows) > 0:
        for i, r in enumerate(rows):
            try:
                r["fold_id"] = int(fold_ids_arr[i])
            except Exception:
                pass
    elif mode == "holdout" and len(rows) > 0:
        for r in rows:
            r["fold_id"] = 1

    payload = {
        "classes": decoder_classes.tolist() if decoder_classes is not None else None,
        "positive_class_label": positive_class_label,
        "positive_class_index": positive_class_index,
        "has_decision_scores": ds_arr is not None,
        "has_proba": pr_arr is not None,
        "summary": summary,
        "notes": notes,
        "n_rows_total": int(np.asarray(y_pred_all).shape[0]) if y_pred_all is not None else 0,
        "preview_rows": rows,
    }

    return DecoderOutputs.model_validate(payload)
