from __future__ import annotations

"""Prediction table builders.

The public surface remains in :mod:`engine.reporting.prediction.prediction_results`.
"""

from typing import Any, Iterable, List, Mapping, Optional

from .coercion import numpy, slice_first_n, to_1d_array


def build_prediction_table(
    *,
    indices: Optional[Iterable[int]],
    y_pred: Any,
    task: str,
    y_true: Optional[Any] = None,
    max_rows: Optional[int] = None,
) -> List[Mapping[str, Any]]:
    """Build a table (list of dicts) summarizing prediction results."""

    np = numpy()

    y_pred_arr = to_1d_array(y_pred)
    y_true_arr = to_1d_array(y_true) if y_true is not None else None

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
    y_pred_arr = slice_first_n(y_pred_arr, n_rows)
    if y_true_arr is not None:
        y_true_arr = slice_first_n(y_true_arr, n_rows)

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
