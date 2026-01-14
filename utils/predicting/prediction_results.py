from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional

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


def _slice_first_n(seq: Any, n: Optional[int]) -> Any:
    if n is None:
        return seq
    if np is not None and isinstance(seq, np.ndarray):
        return seq[:n]
    # Generic slicing for sequences
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

    Parameters
    ----------
    indices:
        Optional iterable of row indices. If None, 0..N-1 is used.
    y_pred:
        Predicted values (labels or regression outputs).
    task:
        "classification" or "regression". Controls what extra columns are added.
    y_true:
        Optional ground-truth labels/targets. If provided, residuals or correctness
        flags are computed.
    max_rows:
        If provided, only the first `max_rows` rows are included in the table.

    Returns
    -------
    List[dict]
        Each row contains at least:
          - "index"
          - "y_pred"
        And optionally:
          - "y_true"
          - For regression: "residual", "abs_error"
          - For classification: "correct" (bool)
    """
    y_pred_arr = _to_1d_array(y_pred)
    y_true_arr = _to_1d_array(y_true) if y_true is not None else None

    # Determine length
    if np is not None and isinstance(y_pred_arr, np.ndarray):
        n = int(y_pred_arr.shape[0])
    else:
        # Try generic len()
        try:
            n = len(y_pred_arr)  # type: ignore[arg-type]
        except Exception:
            # Fall back to single element
            n = 1

    # Normalize indices
    if indices is None:
        idx_list = list(range(n))
    else:
        idx_list = list(indices)
        if len(idx_list) != n:
            # Best effort: fall back to 0..N-1 if lengths disagree
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
            y_pred_val = y_pred_arr[i].item() if hasattr(y_pred_arr[i], "item") else y_pred_arr[i]
        else:
            try:
                y_pred_val = y_pred_arr[i]  # type: ignore[index]
            except Exception:
                y_pred_val = y_pred_arr  # type: ignore[assignment]

        row: dict[str, Any] = {
            "index": int(idx),
            "y_pred": y_pred_val,
        }

        if y_true_arr is not None:
            if np is not None and isinstance(y_true_arr, np.ndarray):
                y_true_val = y_true_arr[i].item() if hasattr(y_true_arr[i], "item") else y_true_arr[i]
            else:
                try:
                    y_true_val = y_true_arr[i]  # type: ignore[index]
                except Exception:
                    y_true_val = y_true_arr  # type: ignore[assignment]

            row["y_true"] = y_true_val

            if task == "regression":
                try:
                    residual = float(y_true_val) - float(y_pred_val)
                    row["residual"] = residual
                    row["abs_error"] = abs(residual)
                except Exception:
                    # If casting fails, just skip residuals
                    pass
            else:
                # classification (or unknown): report correctness flag if comparable
                try:
                    row["correct"] = bool(y_true_val == y_pred_val)
                except Exception:
                    pass

        rows.append(row)

    return rows


def _sanitize_col_suffix(label: Any) -> str:
    """Make a stable, CSV/JSON-friendly column suffix from a class label."""
    s = str(label).strip()
    if not s:
        return "empty"
    # conservative sanitation: spaces to underscores, drop slashes/commas
    for ch in [" ", "/", "\\", ",", ";", ":", "\t", "\n", "\r"]:
        s = s.replace(ch, "_")
    # avoid weird duplicated underscores
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _to_2d_array(x: Any) -> Any:
    """Best-effort coercion of input to a 2D numpy array when numpy is available."""
    if x is None or np is None:
        return x
    try:
        arr = np.asarray(x)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr
    except Exception:
        return x


def build_decoder_output_table(
    *,
    indices: Optional[Iterable[int]],
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
    """Build a per-sample table for decoder outputs (classification only).

    This is designed to be fed into `utils/io/export/result_export.py`.

    Parameters
    ----------
    indices:
        Optional iterable of sample indices. If None, 0..N-1 is used.
    y_pred:
        Hard predictions (shape (n_samples,)).
    classes:
        Optional class ordering corresponding to the columns of `proba` and/or
        2D `decision_scores`.
    decision_scores:
        Output of decision_function(X), if available.
        - binary: (n_samples,)
        - multiclass: (n_samples, n_classes)
    proba:
        Output of predict_proba(X), if available, shape (n_samples, n_classes).
    margin:
        Optional confidence proxy per sample.
    positive_class_label / positive_class_index:
        Optional convenience mapping for a "positive" class. If provided and
        probability/scores exist, adds `positive_proba` and/or `positive_score`.
    y_true:
        Optional ground truth labels. Adds `y_true` and `correct` columns.
    max_rows:
        If set, only the first `max_rows` rows are returned.

    Returns
    -------
    List[dict]
        Rows with columns like:
          - index, y_true, y_pred, correct
          - decoder_score (binary) OR score_<class>
          - p_<class>
          - positive_proba / positive_score
          - margin
    """

    y_pred_arr = _to_1d_array(y_pred)
    y_true_arr = _to_1d_array(y_true) if y_true is not None else None

    ds_arr = _to_2d_array(decision_scores) if decision_scores is not None else None
    pr_arr = _to_2d_array(proba) if proba is not None else None
    mg_arr = _to_1d_array(margin) if margin is not None else None

    cls_arr = None
    if classes is not None:
        cls_arr = _to_1d_array(classes)

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

    # Slice
    n_rows = n if max_rows is None else min(n, max_rows)
    idx_list = idx_list[:n_rows]
    y_pred_arr = _slice_first_n(y_pred_arr, n_rows)
    if y_true_arr is not None:
        y_true_arr = _slice_first_n(y_true_arr, n_rows)
    if ds_arr is not None:
        ds_arr = _slice_first_n(ds_arr, n_rows)
    if pr_arr is not None:
        pr_arr = _slice_first_n(pr_arr, n_rows)
    if mg_arr is not None:
        mg_arr = _slice_first_n(mg_arr, n_rows)

    # If user passed label but not index, resolve index if possible
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

    rows: List[Mapping[str, Any]] = []

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
    if decision_scores is not None:
        try:
            if np is not None:
                ds_tmp = np.asarray(decision_scores)
                ds_is_binary_scalar = ds_tmp.ndim == 1
            else:
                # best effort
                ds_is_binary_scalar = not isinstance(decision_scores, list) or (
                    len(decision_scores) > 0 and not isinstance(decision_scores[0], (list, tuple))
                )
        except Exception:
            ds_is_binary_scalar = False

    for i in range(n_rows):
        idx = idx_list[i]
        y_pred_val = _get_1d_val(y_pred_arr, i)

        row: dict[str, Any] = {
            "index": int(idx),
            "y_pred": y_pred_val,
        }

        if y_true_arr is not None:
            y_true_val = _get_1d_val(y_true_arr, i)
            row["y_true"] = y_true_val
            try:
                row["correct"] = bool(y_true_val == y_pred_val)
            except Exception:
                pass

        # decision scores
        if ds_arr is not None:
            if ds_is_binary_scalar:
                row["decoder_score"] = _get_1d_val(_to_1d_array(decision_scores), i)
            else:
                if score_cols:
                    for j, col in score_cols:
                        row[col] = _get_2d_val(ds_arr, i, j)
                else:
                    # fallback to indexed names
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
            elif ds_arr is not None and ds_is_binary_scalar:
                # for binary, decision_function is already "positive" score
                row["positive_score"] = _get_1d_val(_to_1d_array(decision_scores), i)

        # margin
        if mg_arr is not None:
            row["margin"] = _get_1d_val(mg_arr, i)

        rows.append(row)

    return rows
