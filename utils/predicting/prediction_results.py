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
