from __future__ import annotations

"""Cached evaluation output export use-case.

This mirrors the old backend endpoint /decoder/export.

The runtime cache is populated during training (best-effort) so the user can
export per-sample outputs without rerunning inference.

The backend must not import ``engine.runtime.*``.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from engine.factories.export_factory import make_exporter
from engine.io.export.csv_export import ExportResult

from engine.reporting.prediction.prediction_results import build_prediction_table
from engine.runtime.caches.eval_outputs_cache import eval_outputs_cache


def export_decoder_outputs_to_csv(*, artifact_uid: str, filename: Optional[str] = None) -> ExportResult:
    """Export cached evaluation outputs as CSV (best-effort).

    For supervised artifacts, this exports a prediction-style table:
      index, [fold_id], y_pred, [y_true], [correct]

    For unsupervised artifacts, this exports:
      index, cluster_id, plus any cached per-sample fields.
    """

    cached = eval_outputs_cache.get(artifact_uid)
    if cached is None:
        raise ValueError(
            "No cached evaluation outputs found for this artifact. "
            "Train the model first (or keep the backend process alive) before exporting."
        )

    task = (cached.task or "classification").lower()

    # -----------------
    # Unsupervised export
    # -----------------
    if task == "unsupervised":
        indices = np.asarray(cached.indices) if cached.indices is not None else np.arange(0)
        cluster_id = np.asarray(cached.cluster_id) if cached.cluster_id is not None else np.zeros_like(indices)

        n = int(indices.shape[0])
        table: List[Dict[str, Any]] = []
        for i in range(n):
            row: Dict[str, Any] = {
                "index": int(indices[i]),
                "cluster_id": int(cluster_id[i]) if i < int(cluster_id.shape[0]) else None,
            }
            # Add any cached per-sample fields
            try:
                per = cached.per_sample or {}
                for k, v in per.items():
                    if isinstance(v, (list, tuple, np.ndarray)) and len(v) == n:
                        row[str(k)] = v[i]
            except Exception:
                pass
            table.append(row)

        exporter = make_exporter("csv")
        return exporter.export(table, dest=None, filename=filename)

    # ----------------
    # Supervised export
    # ----------------

    if cached.y_pred is None:
        raise ValueError("Cached evaluation outputs missing y_pred")

    y_pred = np.asarray(cached.y_pred).reshape(-1)
    n = int(y_pred.shape[0])

    idx = np.asarray(cached.indices).reshape(-1) if cached.indices is not None else np.arange(n, dtype=int)
    if int(idx.shape[0]) != n:
        idx = np.arange(n, dtype=int)

    y_true = np.asarray(cached.y_true).reshape(-1) if cached.y_true is not None else None
    if y_true is not None and int(y_true.shape[0]) != n:
        y_true = None

    eval_kind = "regression" if task == "regression" else "classification"

    table = build_prediction_table(
        indices=idx.tolist(),
        y_pred=y_pred,
        y_true=y_true,
        task=eval_kind,
        max_rows=None,
    )

    # fold ids (optional)
    fold_ids = np.asarray(cached.fold_ids).reshape(-1) if cached.fold_ids is not None else None
    if fold_ids is not None and int(fold_ids.shape[0]) == n:
        for i in range(n):
            try:
                table[i]["fold_id"] = int(fold_ids[i])
            except Exception:
                pass

    # Attach any cached per-sample fields (best-effort)
    try:
        per = cached.per_sample or {}
        for k, v in per.items():
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) == n:
                for i in range(n):
                    table[i][str(k)] = v[i]
    except Exception:
        pass

    exporter = make_exporter("csv")
    return exporter.export(table, dest=None, filename=filename)
