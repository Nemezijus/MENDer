from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional
import math
import re

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy may not be installed in all environments
    np = None  # type: ignore[assignment]


def _to_1d_array(x: Any) -> Any:
    """
    Best-effort coercion of input to a 1D numpy array when numpy is available.
    Falls back to the original object if coercion fails.
    """
    if np is None:
        return x
    try:
        arr = np.asarray(x)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr
    except Exception:
        return x


def _to_2d_array(x: Any) -> Any:
    """
    Best-effort coercion of input to a 2D numpy array when numpy is available.
    - scalar -> (1, 1)
    - 1D     -> (n, 1)
    Falls back to the original object if coercion fails.
    """
    if np is None:
        return x
    try:
        arr = np.asarray(x)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr
    except Exception:
        return x


def _slice_first_n(seq: Any, n: Optional[int]) -> Any:
    if n is None:
        return seq
    if np is not None and isinstance(seq, np.ndarray):
        return seq[:n]
    try:
        return seq[:n]
    except Exception:
        return seq


def build_prediction_table(
    *,
    indices: Optional[Iterable[int]],
    y_pred: Any,
    task: str,
    y_true: Optional[Any] = None,
    max_rows: Optional[int] = None,
) -> List[Mapping[str, Any]]:
    """
    Build a table (list of dicts) summarizing prediction results.
    """
    y_pred_arr = _to_1d_array(y_pred)
    y_true_arr = _to_1d_array(y_true) if y_true is not None else None

    # Determine length
    if np is not None and isinstance(y_pred_arr, np.ndarray):
        n = int(y_pred_arr.shape[0])
    else:
        try:
            n = len(y_pred_arr)  # type: ignore[arg-type]
        except Exception:
            n = 1

    # Normalize indices
    if indices is None:
        idx_list = list(range(n))
    else:
        idx_list = list(indices)
        if len(idx_list) != n:
            idx_list = list(range(n))

    # Slice if max_rows is set
    n_rows = n if max_rows is None else min(n, max_rows)
    idx_list = idx_list[:n_rows]
    y_pred_arr = _slice_first_n(y_pred_arr, n_rows)
    if y_true_arr is not None:
        y_true_arr = _slice_first_n(y_true_arr, n_rows)

    rows: List[Mapping[str, Any]] = []

    for i in range(n_rows):
        idx = idx_list[i]

        # Best-effort element extraction
        if np is not None and isinstance(y_pred_arr, np.ndarray):
            v = y_pred_arr[i]
            y_pred_val = v.item() if hasattr(v, "item") else v
        else:
            try:
                y_pred_val = y_pred_arr[i]  # type: ignore[index]
            except Exception:
                y_pred_val = y_pred_arr

        row: dict[str, Any] = {"index": int(idx), "y_pred": y_pred_val}

        if y_true_arr is not None:
            if np is not None and isinstance(y_true_arr, np.ndarray):
                v = y_true_arr[i]
                y_true_val = v.item() if hasattr(v, "item") else v
            else:
                try:
                    y_true_val = y_true_arr[i]  # type: ignore[index]
                except Exception:
                    y_true_val = y_true_arr

            row["y_true"] = y_true_val

            if task == "regression":
                try:
                    residual = float(y_true_val) - float(y_pred_val)
                    row["residual"] = residual
                    row["abs_error"] = abs(residual)
                except Exception:
                    pass
            else:
                try:
                    row["correct"] = bool(y_true_val == y_pred_val)
                except Exception:
                    pass

        rows.append(row)

    return rows


def _sanitize_col_suffix(label: Any) -> str:
    """Make a stable, CSV/JSON-friendly column suffix from a class label."""
    # unwrap numpy scalars
    try:
        if np is not None and isinstance(label, np.generic):
            label = label.item()
    except Exception:
        pass

    # numeric normalization
    try:
        if isinstance(label, bool):
            s = "true" if label else "false"
        elif isinstance(label, int):
            s = str(label)
        elif isinstance(label, float):
            if math.isfinite(label) and abs(label - round(label)) < 1e-9:
                s = str(int(round(label)))
            else:
                s = str(label)
        else:
            s = str(label).strip()
            # normalize int-like float strings (e.g. "-10.0" -> "-10")
            if re.fullmatch(r"-?\d+\.0+", s):
                try:
                    s = str(int(float(s)))
                except Exception:
                    pass
    except Exception:
        s = str(label).strip()

    s = s.strip()
    if not s:
        return "empty"

    for ch in [" ", "/", "\\", ",", ";", ":", "\t", "\n", "\r"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def build_decoder_output_table(
    *,
    indices: Optional[Iterable[int]],
    fold_ids: Optional[Iterable[int]] = None,
    y_pred: Any,
    classes: Optional[Any] = None,
    decision_scores: Optional[Any] = None,
    proba: Optional[Any] = None,
    margin: Optional[Any] = None,
    positive_class_label: Optional[Any] = None,
    positive_class_index: Optional[int] = None,
    y_true: Optional[Any] = None,
    max_rows: Optional[int] = None,
) -> List[Mapping[str, Any]]:
    """Build a per-sample table for decoder outputs (classification only)."""

    y_pred_arr = _to_1d_array(y_pred)
    y_true_arr = _to_1d_array(y_true) if y_true is not None else None

    ds_arr = _to_2d_array(decision_scores) if decision_scores is not None else None
    pr_arr = _to_2d_array(proba) if proba is not None else None
    mg_arr = _to_1d_array(margin) if margin is not None else None

    cls_arr = _to_1d_array(classes) if classes is not None else None

    # Determine length
    if np is not None and isinstance(y_pred_arr, np.ndarray):
        n = int(y_pred_arr.shape[0])
    else:
        try:
            n = len(y_pred_arr)  # type: ignore[arg-type]
        except Exception:
            n = 1

    # Normalize indices
    if indices is None:
        idx_list = list(range(n))
    else:
        idx_list = list(indices)
        if len(idx_list) != n:
            idx_list = list(range(n))

    # Normalize fold ids (optional)
    fold_list: Optional[List[int]] = None
    if fold_ids is not None:
        try:
            fold_list = [int(x) for x in fold_ids]
            if len(fold_list) != n:
                fold_list = None
        except Exception:
            fold_list = None

    # Slice
    n_rows = n if max_rows is None else min(n, max_rows)
    idx_list = idx_list[:n_rows]
    if fold_list is not None:
        fold_list = fold_list[:n_rows]

    y_pred_arr = _slice_first_n(y_pred_arr, n_rows)
    if y_true_arr is not None:
        y_true_arr = _slice_first_n(y_true_arr, n_rows)
    if ds_arr is not None:
        ds_arr = _slice_first_n(ds_arr, n_rows)
    if pr_arr is not None:
        pr_arr = _slice_first_n(pr_arr, n_rows)
    if mg_arr is not None:
        mg_arr = _slice_first_n(mg_arr, n_rows)

    # Resolve positive class index if label provided
    if positive_class_index is None and positive_class_label is not None and cls_arr is not None:
        try:
            if np is not None and isinstance(cls_arr, np.ndarray):
                matches = np.where(cls_arr == positive_class_label)[0]
                if matches.size:
                    positive_class_index = int(matches[0])
            else:
                positive_class_index = list(cls_arr).index(positive_class_label)  # type: ignore[arg-type]
        except Exception:
            positive_class_index = None

    # Build per-class column names
    score_cols: list[tuple[int, str]] = []
    proba_cols: list[tuple[int, str]] = []
    if cls_arr is not None:
        try:
            labels = list(cls_arr)  # type: ignore[arg-type]
        except Exception:
            labels = []
        for j, lab in enumerate(labels):
            suf = _sanitize_col_suffix(lab)
            score_cols.append((j, f"score_{suf}"))
            proba_cols.append((j, f"p_{suf}"))

    def _get_1d_val(arr: Any, i: int) -> Any:
        if np is not None and isinstance(arr, np.ndarray):
            v = arr[i]
            return v.item() if hasattr(v, "item") else v
        try:
            return arr[i]  # type: ignore[index]
        except Exception:
            return arr

    def _get_2d_val(arr: Any, i: int, j: int) -> Any:
        if np is not None and isinstance(arr, np.ndarray):
            v = arr[i, j]
            return v.item() if hasattr(v, "item") else v
        try:
            return arr[i][j]  # type: ignore[index]
        except Exception:
            return None

    # Determine if decision_scores were binary 1D originally
    ds_is_binary_scalar = False
    ds_1d_arr = None
    if decision_scores is not None:
        try:
            if np is not None:
                ds_tmp = np.asarray(decision_scores)
                ds_is_binary_scalar = ds_tmp.ndim == 1
                if ds_is_binary_scalar:
                    ds_1d_arr = _to_1d_array(decision_scores)
            else:
                ds_is_binary_scalar = (
                    not isinstance(decision_scores, list)
                    or (len(decision_scores) > 0 and not isinstance(decision_scores[0], (list, tuple)))
                )
                if ds_is_binary_scalar:
                    ds_1d_arr = _to_1d_array(decision_scores)
        except Exception:
            ds_is_binary_scalar = False
            ds_1d_arr = None

    rows: List[Mapping[str, Any]] = []

    for i in range(n_rows):
        idx = idx_list[i]
        y_pred_val = _get_1d_val(y_pred_arr, i)

        row: dict[str, Any] = {"index": int(idx), "y_pred": y_pred_val}

        if fold_list is not None:
            row["fold_id"] = fold_list[i]

        if y_true_arr is not None:
            y_true_val = _get_1d_val(y_true_arr, i)
            row["y_true"] = y_true_val
            try:
                row["correct"] = bool(y_true_val == y_pred_val)
            except Exception:
                pass

        # decision scores
        if ds_arr is not None:
            if ds_is_binary_scalar and ds_1d_arr is not None:
                row["decoder_score"] = _get_1d_val(ds_1d_arr, i)
            else:
                if score_cols:
                    for j, col in score_cols:
                        row[col] = _get_2d_val(ds_arr, i, j)
                else:
                    try:
                        ncol = int(ds_arr.shape[1]) if np is not None and isinstance(ds_arr, np.ndarray) else len(ds_arr[i])
                        for j in range(ncol):
                            row[f"score_{j}"] = _get_2d_val(ds_arr, i, j)
                    except Exception:
                        pass

        # probabilities
        if pr_arr is not None:
            if proba_cols:
                for j, col in proba_cols:
                    row[col] = _get_2d_val(pr_arr, i, j)
            else:
                try:
                    ncol = int(pr_arr.shape[1]) if np is not None and isinstance(pr_arr, np.ndarray) else len(pr_arr[i])
                    for j in range(ncol):
                        row[f"p_{j}"] = _get_2d_val(pr_arr, i, j)
                except Exception:
                    pass

        # positive class convenience outputs
        if positive_class_index is not None:
            j = positive_class_index
            if pr_arr is not None:
                v = _get_2d_val(pr_arr, i, j)
                if v is not None:
                    row["positive_proba"] = v
            if ds_arr is not None and not ds_is_binary_scalar:
                v = _get_2d_val(ds_arr, i, j)
                if v is not None:
                    row["positive_score"] = v
            elif ds_is_binary_scalar and ds_1d_arr is not None:
                row["positive_score"] = _get_1d_val(ds_1d_arr, i)

        # margin
        if mg_arr is not None:
            row["margin"] = _get_1d_val(mg_arr, i)

        rows.append(row)

    return rows


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
