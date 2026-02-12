from __future__ import annotations

"""Small helpers for combining per-sample tables."""

from typing import Any, List, Mapping

from engine.reporting.common.json_safety import ReportError, error_row_from_errors


def merge_prediction_and_decoder_tables(
    *,
    prediction_rows: List[Mapping[str, Any]],
    decoder_rows: List[Mapping[str, Any]],
    index_key: str = "index",
    overwrite: bool = False,
) -> List[Mapping[str, Any]]:
    """Merge per-sample prediction rows with decoder-output rows by `index`."""

    # Preserve explicit error rows (no merging performed on them).
    pred_errors: List[ReportError] = []
    dec_errors: List[ReportError] = []

    pred_ok: List[Mapping[str, Any]] = []
    for r in prediction_rows:
        try:
            if bool(r.get("__error__")):
                pred_errors.extend(list(r.get("errors") or []))  # type: ignore[arg-type]
            else:
                pred_ok.append(r)
        except Exception:
            pred_ok.append(r)

    dec_ok: List[Mapping[str, Any]] = []
    for r in decoder_rows:
        try:
            if bool(r.get("__error__")):
                dec_errors.extend(list(r.get("errors") or []))  # type: ignore[arg-type]
            else:
                dec_ok.append(r)
        except Exception:
            dec_ok.append(r)

    dec_by_idx: dict[Any, Mapping[str, Any]] = {}
    for r in dec_ok:
        try:
            dec_by_idx[r.get(index_key)] = r
        except Exception:
            continue

    merged: List[Mapping[str, Any]] = []
    for prow in pred_ok:
        idx = prow.get(index_key)
        drow = dec_by_idx.get(idx)
        if not drow:
            merged.append(prow)
            continue

        out: dict[str, Any] = dict(prow)
        for k, v in drow.items():
            if k == index_key:
                continue
            if not overwrite and k in out:
                continue
            out[k] = v
        merged.append(out)

    all_errors: List[ReportError] = []
    if pred_errors:
        all_errors.extend(pred_errors)
    if dec_errors:
        all_errors.extend(dec_errors)

    if all_errors:
        merged.append(error_row_from_errors(all_errors))
    return merged
