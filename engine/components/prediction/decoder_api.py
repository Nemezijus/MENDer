from __future__ import annotations

"""Canonical prediction/decoder API.

Segment 6 goal: consolidate all decoder-output generation through ONE path.

This module turns raw per-sample decoder arrays (scores/probabilities/margins)
into the *result contract* used by the rest of MENDer:
`engine.contracts.results.decoder.DecoderOutputs`.

Compute (raw extraction) lives in:
    engine.components.prediction.decoder_extraction

Reporting utilities used here:
    engine.reporting.prediction.prediction_results.build_decoder_output_table
    engine.reporting.decoder.decoder_summaries.compute_decoder_summaries

The public entrypoint is `predict_decoder_outputs(...)`.
"""

from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from engine.contracts.results.decoder import DecoderOutputs

from .decoder_extraction import RawDecoderOutputs, compute_decoder_outputs_raw

from engine.reporting.prediction.prediction_results import build_decoder_output_table
from engine.reporting.decoder.decoder_summaries import compute_decoder_summaries


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for s in items:
        ss = str(s)
        if ss not in seen:
            seen.add(ss)
            out.append(ss)
    return out


def _as_1d(x: Any) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 0:
        return a.reshape(1)
    return a.reshape(-1)


def _as_2d_or_none(x: Any | None) -> Optional[np.ndarray]:
    if x is None:
        return None
    a = np.asarray(x)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    raise ValueError(f"Expected 2D array; got shape {a.shape}.")


def _ensure_indices(indices: Optional[Iterable[int]], n: int) -> List[int]:
    if indices is None:
        return list(range(int(n)))
    try:
        idx = [int(i) for i in list(indices)]
    except Exception:
        return list(range(int(n)))
    if len(idx) != int(n):
        return list(range(int(n)))
    return idx


def _normalize_classes(classes: Any | None) -> Optional[List[Any]]:
    if classes is None:
        return None
    try:
        arr = np.asarray(classes)
        return [c.item() if isinstance(c, np.generic) else c for c in arr.tolist()]
    except Exception:
        try:
            return list(classes)  # type: ignore[arg-type]
        except Exception:
            return None


def _compute_margin(
    *,
    decision_scores: Optional[np.ndarray],
    proba: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Best-effort confidence proxy.

    - Prefer decision scores when present.
    - Fall back to probabilities/vote shares.
    """

    if decision_scores is not None:
        ds = np.asarray(decision_scores)
        if ds.ndim == 1:
            return np.abs(ds)
        if ds.ndim == 2 and ds.shape[1] >= 2:
            part = np.partition(ds, kth=-2, axis=1)
            top2 = part[:, -2]
            top1 = part[:, -1]
            return top1 - top2
        return None

    if proba is not None:
        p = np.asarray(proba)
        if p.ndim == 2 and p.shape[1] >= 2:
            if p.shape[1] == 2:
                return np.abs(p[:, 1] - p[:, 0])
            part = np.partition(p, kth=-2, axis=1)
            return part[:, -1] - part[:, -2]
    return None


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
    """Compute decoder outputs using the canonical engine path.

    Returns the *result contract* `DecoderOutputs`.

    Notes
    -----
    - `max_preview_rows=None` produces full rows (useful for export).
    - Summary can be skipped for cheap preview.
    """

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
    idx_list = _ensure_indices(indices, n)

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
        y_pred=_as_1d(raw.y_pred),
        y_true=_as_1d(y_true) if y_true is not None else None,
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

    payload = {
        "classes": _normalize_classes(raw.classes),
        "positive_class_label": raw.positive_class_label,
        "positive_class_index": raw.positive_class_index,
        "has_decision_scores": bool(ds is not None),
        "has_proba": bool(pr is not None),
        "proba_source": raw.proba_source,
        "notes": _dedupe_preserve_order(notes),
        "preview_rows": rows,
        "n_rows_total": int(n),
        "summary": summary,
    }

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

    y_pred_arr = _as_1d(y_pred)
    n = int(y_pred_arr.shape[0])

    idx_list = _ensure_indices(indices, n)

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
    pr_arr = _as_2d_or_none(proba)
    mg_arr = np.asarray(margin) if margin is not None else None
    if mg_arr is None:
        mg_arr = _compute_margin(decision_scores=ds_arr, proba=pr_arr)

    rows = build_decoder_output_table(
        indices=idx_list,
        fold_ids=fold_ids_list,
        y_pred=y_pred_arr,
        y_true=_as_1d(y_true) if y_true is not None else None,
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

    payload = {
        "classes": _normalize_classes(classes_arr),
        "positive_class_label": positive_class_label,
        "positive_class_index": pos_idx,
        "has_decision_scores": bool(ds_arr is not None),
        "has_proba": bool(pr_arr is not None),
        "proba_source": proba_source,
        "notes": _dedupe_preserve_order(all_notes),
        "preview_rows": rows,
        "n_rows_total": int(n),
        "summary": summary,
    }

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
    """Train-service helper: build decoder outputs from fold parts.

    This is the canonical replacement for backend-local payload assembly.
    """

    def _concat_if_complete(parts: List[Optional[np.ndarray]], name: str) -> Optional[np.ndarray]:
        if not parts:
            return None
        if any(p is None for p in parts):
            notes.append(f"Decoder outputs: '{name}' omitted because at least one fold could not produce it.")
            return None
        try:
            return np.concatenate([np.asarray(p) for p in parts], axis=0)
        except Exception as e:
            notes.append(f"Decoder outputs: '{name}' omitted due to concat error ({type(e).__name__}: {e}).")
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

    # For holdout, set fold_id=1 to match prior UI expectations.
    if fold_ids_arr is None and str(mode).lower() == "holdout":
        try:
            fold_ids_arr = np.ones((int(np.asarray(y_pred_all).shape[0]),), dtype=int)
        except Exception:
            fold_ids_arr = None

    max_rows = int(getattr(decoder_cfg, "max_preview_rows", 200)) if decoder_cfg is not None else 200

    indices_out = [int(v) for v in (row_indices.tolist() if row_indices is not None else [])]

    # Build via the canonical array-based builder
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

    # Add concise summary notes (best-effort) to match historical behavior.
    try:
        summ = dec.summary if isinstance(dec.summary, dict) else None
        if summ:
            bits = []
            if "log_loss" in summ:
                bits.append(f"log_loss={float(summ['log_loss']):.4f}")
            if "brier" in summ:
                bits.append(f"brier={float(summ['brier']):.4f}")
            if "margin_mean" in summ:
                bits.append(f"margin_mean={float(summ['margin_mean']):.4f}")
            if "max_proba_mean" in summ:
                bits.append(f"mean_max_proba={float(summ['max_proba_mean']):.4f}")
            if bits:
                # Extend notes on the returned contract (without duplicating too aggressively)
                extra = "Decoder summary: " + ", ".join(bits)
                if extra not in dec.notes:
                    dec.notes.append(extra)
    except Exception:
        pass

    return dec
