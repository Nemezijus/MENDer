from __future__ import annotations

"""Small helpers for combining per-sample tables."""

from typing import Any, List, Mapping


def merge_prediction_and_decoder_tables(
    *,
    prediction_rows: List[Mapping[str, Any]],
    decoder_rows: List[Mapping[str, Any]],
    index_key: str = "index",
    overwrite: bool = False,
) -> List[Mapping[str, Any]]:
    """Merge per-sample prediction rows with decoder-output rows by `index`."""

    dec_by_idx: dict[Any, Mapping[str, Any]] = {}
    for r in decoder_rows:
        try:
            dec_by_idx[r.get(index_key)] = r
        except Exception:
            continue

    merged: List[Mapping[str, Any]] = []
    for prow in prediction_rows:
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

    return merged
