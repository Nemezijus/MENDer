from __future__ import annotations

"""Formatting helpers for decoder outputs."""

from typing import Any, Dict, List, Optional

import numpy as np

from .inputs import dedupe_preserve_order, normalize_classes


def build_decoder_outputs_payload(
    *,
    classes: Any | None,
    positive_class_label: Any | None,
    positive_class_index: int | None,
    has_decision_scores: bool,
    has_proba: bool,
    proba_source: str | None,
    notes: List[str],
    preview_rows: List[Dict[str, Any]],
    n_rows_total: int,
    summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create the dict passed into DecoderOutputs.model_validate."""

    return {
        "classes": normalize_classes(classes),
        "positive_class_label": positive_class_label,
        "positive_class_index": positive_class_index,
        "has_decision_scores": bool(has_decision_scores),
        "has_proba": bool(has_proba),
        "proba_source": proba_source,
        "notes": dedupe_preserve_order([str(n) for n in notes]),
        "preview_rows": preview_rows,
        "n_rows_total": int(n_rows_total),
        "summary": summary,
    }


def append_concise_summary_notes(notes: List[str], summary: Any) -> None:
    """Best-effort short notes derived from a decoder summary."""

    try:
        summ = summary if isinstance(summary, dict) else None
        if not summ:
            return
        bits: List[str] = []
        if "log_loss" in summ:
            bits.append(f"log_loss={float(summ['log_loss']):.4f}")
        if "brier_score" in summ:
            bits.append(f"brier={float(summ['brier_score']):.4f}")
        if "roc_auc" in summ:
            bits.append(f"roc_auc={float(summ['roc_auc']):.4f}")
        if bits:
            notes.append("Decoder summary: " + ", ".join(bits))
    except Exception:
        return