from __future__ import annotations

"""Unsupervised (clustering) training orchestration (Engine use-case).

This is the BL-layer counterpart of the historical backend implementation in
``backend/app/services/train_service.py``.
"""

from typing import Any, Dict, Optional

import numpy as np

from engine.contracts.unsupervised_configs import UnsupervisedRunConfig
from engine.contracts.results.unsupervised import UnsupervisedResult
from engine.io.artifacts.meta import ArtifactBuilderInput, build_model_artifact_meta
from engine.io.artifacts.store import ArtifactStore
from engine.reporting.common.json_safety import safe_float_optional
from engine.runtime.caches.artifact_cache import artifact_cache
from engine.runtime.caches.eval_outputs_cache import EvalOutputs, eval_outputs_cache
from engine.use_cases._deps import resolve_seed, resolve_store
from engine.use_cases.artifacts import save_model_to_store

from engine.factories.data_loading_factory import make_data_loader
from engine.factories.eval_factory import make_unsupervised_evaluator
from engine.factories.pipeline_factory import make_unsupervised_pipeline
from engine.runtime.random.rng import RngManager


def train_unsupervised(
    cfg: UnsupervisedRunConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> UnsupervisedResult:
    """Train an unsupervised (clustering) model.

    Notes
    -----
    - ``y`` may exist in the input data but is ignored.
    - Some estimators do not support ``predict`` on unseen data; apply handling
      is best-effort and warnings are surfaced.
    """

    # --- Load training data -------------------------------------------------
    loader = make_data_loader(cfg.data)
    X, _y_ignored = loader.load()

    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError(
            "Unsupervised training requires a 2D feature matrix X with at least 1 sample and 1 feature."
        )

    # --- RNG ----------------------------------------------------------------
    seed = resolve_seed(getattr(cfg.eval, "seed", None), fallback=0) if rng is None else int(rng)
    rngm = RngManager(seed)

    # --- Build + fit pipeline ----------------------------------------------
    pipeline = make_unsupervised_pipeline(cfg, rngm, stream="unsupervised/train")
    pipeline.fit(X)

    # --- Extract training labels -------------------------------------------
    labels: Optional[np.ndarray] = None
    try:
        est = pipeline.steps[-1][1] if hasattr(pipeline, "steps") and pipeline.steps else pipeline
    except Exception:
        est = pipeline

    lab = getattr(est, "labels_", None)
    if lab is not None:
        labels = np.asarray(lab)

    if labels is None and hasattr(pipeline, "predict"):
        try:
            labels = np.asarray(pipeline.predict(X))
        except Exception:
            labels = None

    if labels is None:
        raise ValueError(
            f"Could not obtain cluster labels from estimator {type(est).__name__}. "
            "Expected attribute 'labels_' or method 'predict'."
        )

    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != X.shape[0]:
        raise ValueError("Cluster labels length does not match number of samples in X.")

    # --- Compute unsupervised diagnostics ----------------------------------
    evaluator = make_unsupervised_evaluator(cfg.eval)
    diag = evaluator.evaluate(X, labels, model=pipeline)

    # --- Optional apply/predict on unseen data -----------------------------
    n_apply: Optional[int] = None
    apply_notes: list[str] = []
    if cfg.apply is not None and cfg.fit_scope == "train_and_predict":
        try:
            loader2 = make_data_loader(cfg.apply)
            X_apply, _y2_ignored = loader2.load()
            X_apply = np.asarray(X_apply)
            n_apply = int(X_apply.shape[0])
            if hasattr(pipeline, "predict"):
                _ = pipeline.predict(X_apply)
            else:
                apply_notes.append(
                    f"apply requested but estimator {type(est).__name__} does not support predict(); skipping apply."
                )
        except Exception as e:
            apply_notes.append(f"apply failed: {type(e).__name__}: {e}")

    # --- Build per-sample preview table ------------------------------------
    per_sample = diag.get("per_sample") or {}
    n_rows_total = int(X.shape[0])
    preview_n = min(50, n_rows_total)

    preview_rows: list[Dict[str, Any]] = []
    cluster_ids = per_sample.get("cluster_id")
    if cluster_ids is None:
        cluster_ids = [int(v) for v in labels.tolist()]
        per_sample["cluster_id"] = cluster_ids

    for i in range(preview_n):
        row: Dict[str, Any] = {"index": int(i), "cluster_id": int(cluster_ids[i])}
        for k, v in per_sample.items():
            if k in ("cluster_id",):
                continue
            try:
                row[k] = v[i]
            except Exception:
                continue
        preview_rows.append(row)

    # --- Artifact meta + caching -------------------------------------------
    metrics_dict = diag.get("metrics") or {}
    primary_metric_name = None
    primary_metric_value = None
    if isinstance(metrics_dict, dict):
        if metrics_dict.get("silhouette") is not None:
            primary_metric_name = "silhouette"
            primary_metric_value = safe_float_optional(metrics_dict.get("silhouette"))
        else:
            for k, v in metrics_dict.items():
                if v is not None:
                    primary_metric_name = str(k)
                    primary_metric_value = safe_float_optional(v)
                    break

    n_features_in = int(X.shape[1])
    artifact_input = ArtifactBuilderInput(
        cfg=cfg,
        pipeline=pipeline,
        n_train=int(X.shape[0]),
        n_test=None,
        n_features=n_features_in,
        classes=None,
        kind="unsupervised",
        summary={
            "metric_name": primary_metric_name,
            "metric_value": primary_metric_value,
            "mean_score": None,
            "std_score": None,
            "n_splits": None,
            "notes": [],
            "extra_stats": {
                "unsupervised_metrics": metrics_dict,
                "cluster_summary": diag.get("cluster_summary"),
            },
        },
    )
    artifact_meta = build_model_artifact_meta(artifact_input)

    # Persist artifact bytes (default store if not provided)
    store_resolved = resolve_store(store)
    try:
        save_model_to_store(store_resolved, pipeline, artifact_meta)
    except Exception as e:
        raise RuntimeError(
            f"Persisting unsupervised model artifact failed (uid={artifact_meta.get('uid')!r})."
        ) from e

    try:
        uid = artifact_meta["uid"]
        artifact_cache.put(uid, pipeline)
        try:
            eval_outputs_cache.put(
                uid,
                EvalOutputs(
                    task="unsupervised",
                    indices=np.arange(int(X.shape[0]), dtype=int),
                    cluster_id=np.asarray(labels, dtype=int),
                    per_sample=per_sample,
                ),
            )
        except Exception as e:
            apply_notes.append(f"eval_outputs_cache.put failed: {type(e).__name__}: {e}")
    except Exception as e:
        apply_notes.append(f"artifact_cache.put failed: {type(e).__name__}: {e}")

    # --- Assemble response --------------------------------------------------
    out_metrics: Dict[str, Optional[float]] = {}
    if isinstance(metrics_dict, dict):
        for k, v in metrics_dict.items():
            out_metrics[str(k)] = safe_float_optional(v)

    warnings_all: list[str] = []
    try:
        warnings_all.extend([str(x) for x in (diag.get("warnings") or [])])
    except Exception as e:
        apply_notes.append(f"artifact_cache.put failed: {type(e).__name__}: {e}")
    warnings_all.extend(apply_notes)

    out = {
        "task": "unsupervised",
        "n_train": int(X.shape[0]),
        "n_features": n_features_in,
        "n_apply": n_apply,
        "metrics": out_metrics,
        "warnings": warnings_all,
        "cluster_summary": diag.get("cluster_summary") or {},
        "diagnostics": {
            "model_diagnostics": diag.get("model_diagnostics") or {},
            "embedding_2d": diag.get("embedding_2d"),
            "plot_data": diag.get("plot_data") or {},
        },
        "artifact": artifact_meta,
        "unsupervised_outputs": {
            "notes": [],
            "preview_rows": preview_rows,
            "n_rows_total": n_rows_total,
            "summary": {
                "n_clusters": (diag.get("cluster_summary") or {}).get("n_clusters"),
                "n_noise": (diag.get("cluster_summary") or {}).get("n_noise"),
            },
        },
        "notes": [],
    }

    return UnsupervisedResult.model_validate(out)
