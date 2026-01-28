import numpy as np
from typing import Dict, Any, Optional

from shared_schemas.run_config import RunConfig
from shared_schemas.unsupervised_configs import UnsupervisedRunConfig
from shared_schemas.model_configs import get_model_task
from utils.factories.data_loading_factory import make_data_loader
from utils.factories.sanity_factory import make_sanity_checker
from utils.factories.split_factory import make_splitter
from utils.factories.pipeline_factory import make_pipeline, make_unsupervised_pipeline
from utils.factories.eval_factory import make_evaluator, make_unsupervised_evaluator
from utils.factories.metrics_factory import make_metrics_computer
from utils.permutations.rng import RngManager
from utils.factories.baseline_factory import make_baseline
from utils.persistence.modelArtifact import ArtifactBuilderInput, build_model_artifact_meta
from utils.persistence.artifact_cache import artifact_cache
from utils.persistence.eval_outputs_cache import EvalOutputs, eval_outputs_cache
from utils.postprocessing.scoring import PROBA_METRICS
from utils.postprocessing.decoder_outputs import compute_decoder_outputs

from ..adapters.io_adapter import LoadError
from ..progress.registry import PROGRESS

from .common.json_safety import safe_float_optional
from .training.metrics_payloads import normalize_confusion, normalize_roc
from .training.decoder_payloads import build_decoder_outputs_payload
from .training.regression_payloads import (
    build_regression_diagnostics_payload,
    build_regression_decoder_outputs_payload,
)


def train(cfg: RunConfig) -> Dict[str, Any]:
    # --- Load data -----------------------------------------------------------
    try:
        loader = make_data_loader(cfg.data)
        X, y = loader.load()
        if y is None:
            raise LoadError("Supervised training requires both X and y. For X-only data, use Unsupervised learning.")
    except Exception as e:
        raise LoadError(str(e))

    # --- Checks --------------------------------------------------------------
    make_sanity_checker().check(X, y)

    # --- RNG -----------------------------------------------------------------
    rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

    # --- Split (hold-out or k-fold) -----------------------------------------
    split_seed = rngm.child_seed("train/split")

    try:
        mode = getattr(cfg.split, "mode")
    except AttributeError:
        raise ValueError("RunConfig.split is missing required 'mode' attribute")

    if mode not in ("holdout", "kfold"):
        raise ValueError(f"Unsupported split mode in train service: {mode!r}")

    # Decide evaluation kind based on model config (classification vs regression)
    task = get_model_task(cfg.model)
    eval_kind = "regression" if task == "regression" else "classification"

    metrics_computer = make_metrics_computer(kind=eval_kind)
    # For regression, stratification does not make sense -> force it off defensively
    if eval_kind == "regression" and hasattr(cfg.split, "stratified"):
        cfg.split.stratified = False

    splitter = make_splitter(cfg.split, seed=split_seed)

    # --- Fit / evaluate over splits (handles both holdout & kfold) ----------
    evaluator = make_evaluator(cfg.eval, kind=eval_kind)

    fold_scores = []
    y_true_all, y_pred_all = [], []
    y_proba_all, y_score_all = [], []
    # If the splitter provides original sample indices, keep them so decoder outputs
    # can be shown in the original row order (useful for downstream alignment).
    test_indices_parts: list[np.ndarray | None] = []
    # Track fold id for each evaluation row (useful for CV exports / diagnostics).
    eval_fold_ids_parts: list[np.ndarray] = []
    n_train_sizes, n_test_sizes = [], []

    # --- Optional decoder outputs (classification only) ---------------------
    decoder_cfg = getattr(cfg.eval, "decoder", None)
    decoder_enabled = bool(getattr(decoder_cfg, "enabled", False)) if decoder_cfg is not None else False
    decoder_positive_label = getattr(decoder_cfg, "positive_class_label", None) if decoder_cfg is not None else None
    decoder_include_scores = bool(getattr(decoder_cfg, "include_decision_scores", True)) if decoder_cfg is not None else True
    decoder_include_probabilities = bool(getattr(decoder_cfg, "include_probabilities", True)) if decoder_cfg is not None else True
    decoder_calibrate_probabilities = bool(getattr(decoder_cfg, "calibrate_probabilities", False)) if decoder_cfg is not None else False

    # For k-fold CV, decoder outputs should be *out-of-fold* (OOF):
    # compute them with the fold-trained model on that fold's test split.
    # We keep per-fold decoder parts so we can safely drop a field (scores/proba/margin)
    # if any fold cannot produce it.
    decoder_scores_parts: list[np.ndarray | None] = []
    decoder_proba_parts: list[np.ndarray | None] = []
    decoder_margin_parts: list[np.ndarray | None] = []
    decoder_notes_all: list[str] = []
    decoder_classes: np.ndarray | None = None
    decoder_positive_index: int | None = None
    decoder_fold_ids: list[np.ndarray] = []

    for fold_id, split in enumerate(splitter.split(X, y), start=1):
        # Support splitters that optionally yield indices.
        # Expected formats:
        #   (Xtr, Xte, ytr, yte)
        #   (idx_tr, idx_te, Xtr, Xte, ytr, yte)
        if isinstance(split, (tuple, list)) and len(split) == 4:
            Xtr, Xte, ytr, yte = split
            idx_te = None
        elif isinstance(split, (tuple, list)) and len(split) == 6:
            _, idx_te, Xtr, Xte, ytr, yte = split
        else:
            raise ValueError(
                "Splitter yielded an unsupported split tuple; expected 4 or 6 items. "
                f"Got {type(split).__name__} of length {len(split) if isinstance(split, (tuple, list)) else 'n/a'}."
            )

        pipeline = make_pipeline(cfg, rngm, stream=f"{mode}/fold{fold_id}")
        pipeline.fit(Xtr, ytr)
        y_pred = pipeline.predict(Xte)
        last_pipeline = pipeline

        # Keep test indices for optional re-ordering back to original row order.
        if idx_te is not None:
            test_indices_parts.append(np.asarray(idx_te, dtype=int).ravel())
        else:
            test_indices_parts.append(None)

        # Always keep fold_id per row (works for both holdout and k-fold).
        try:
            n_fold_rows = int(np.asarray(y_pred).shape[0])
        except Exception:
            n_fold_rows = int(Xte.shape[0])
        eval_fold_ids_parts.append(np.full((n_fold_rows,), fold_id, dtype=int))

        metric_name = cfg.eval.metric
        y_proba = None
        y_score = None

        # For classification, we try to compute scores/probabilities once per fold.
        if eval_kind == "classification":
            # Try to compute both proba and scores when available.
            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(Xte)
            if hasattr(pipeline, "decision_function"):
                y_score = pipeline.decision_function(Xte)

            # If the chosen metric *needs* probabilities/scores, enforce availability
            if metric_name in PROBA_METRICS and y_proba is None and y_score is None:
                raise ValueError(
                    f"Metric '{metric_name}' requires predict_proba or decision_function, "
                    f"but estimator {type(pipeline).__name__} has neither."
                )
        score_val = evaluator.score(
            yte,
            y_pred=y_pred,
            y_proba=y_proba,
            y_score=y_score,
        )
        fold_scores.append(float(score_val))

        y_true_all.append(yte)
        y_pred_all.append(y_pred)
        if y_proba is not None:
            y_proba_all.append(y_proba)
        if y_score is not None:
            y_score_all.append(y_score)

        if decoder_enabled and eval_kind == "classification":
            # Track fold ID per test row for UI/diagnostics.
            decoder_fold_ids.append(np.full((n_fold_rows,), fold_id, dtype=int))

            # Compute decoder outputs per fold (OOF) so CV diagnostics are not optimistic.
            try:
                dec = compute_decoder_outputs(
                    pipeline,
                    Xte,
                    positive_class_label=decoder_positive_label,
                    include_decision_scores=decoder_include_scores,
                    include_probabilities=decoder_include_probabilities,
                    calibrate_probabilities=decoder_calibrate_probabilities,
                )

                if decoder_classes is None and dec.classes is not None:
                    decoder_classes = np.asarray(dec.classes)
                if decoder_positive_index is None and dec.positive_class_index is not None:
                    decoder_positive_index = int(dec.positive_class_index)

                decoder_scores_parts.append(
                    np.asarray(dec.decision_scores) if dec.decision_scores is not None else None
                )
                decoder_proba_parts.append(
                    np.asarray(dec.proba) if dec.proba is not None else None
                )
                decoder_margin_parts.append(
                    np.asarray(dec.margin) if dec.margin is not None else None
                )

                if dec.notes:
                    decoder_notes_all.extend([str(x) for x in dec.notes])
            except Exception as e:
                # Keep row alignment: still append None placeholders for this fold.
                decoder_scores_parts.append(None)
                decoder_proba_parts.append(None)
                decoder_margin_parts.append(None)
                decoder_notes_all.append(
                    f"decoder outputs failed on fold {fold_id}: {type(e).__name__}: {e}"
                )
        n_train_sizes.append(int(Xtr.shape[0]))
        n_test_sizes.append(int(Xte.shape[0]))

    refit_pipeline = None
    if mode == "kfold":
        refit_pipeline = make_pipeline(cfg, rngm, stream="kfold/refit")
        refit_pipeline.fit(X, y)

    effective_pipeline = refit_pipeline if refit_pipeline is not None else last_pipeline

    # --- Aggregate scores and metrics (confusion + ROC) ---------------------
    if not fold_scores:
        fold_scores = [float("nan")]

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))  # parity with CV

    # Concatenate per-fold arrays
    y_true_all_arr = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred_all_arr = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    fold_ids_all_arr = np.concatenate(eval_fold_ids_parts) if eval_fold_ids_parts else None

    # If we have original indices from the splitter, reorder OOF arrays back to the
    # original row order. This makes decoder outputs align with the user's input.
    order = None
    row_indices_arr: np.ndarray | None = None
    if test_indices_parts and all(p is not None for p in test_indices_parts):
        idx_all_arr = np.concatenate([np.asarray(p) for p in test_indices_parts], axis=0)
        # Only reorder if lengths match (defensive in case a splitter mixes formats).
        if idx_all_arr.shape[0] == y_pred_all_arr.shape[0]:
            order = np.argsort(idx_all_arr, kind="stable")
            row_indices_arr = idx_all_arr[order]
            y_true_all_arr = y_true_all_arr[order]
            y_pred_all_arr = y_pred_all_arr[order]
            if fold_ids_all_arr is not None and fold_ids_all_arr.shape[0] == order.shape[0]:
                fold_ids_all_arr = fold_ids_all_arr[order]
            if fold_ids_all_arr is not None and fold_ids_all_arr.shape[0] == order.shape[0]:
                fold_ids_all_arr = fold_ids_all_arr[order]
    if row_indices_arr is None:
        row_indices_arr = np.arange(int(y_pred_all_arr.shape[0]), dtype=int)

    y_proba_all_arr = np.concatenate(y_proba_all, axis=0) if y_proba_all else None
    y_score_all_arr = np.concatenate(y_score_all, axis=0) if y_score_all else None

    # Apply the same ordering to proba/scores if they are present and aligned.
    if order is not None:
        if y_proba_all_arr is not None and y_proba_all_arr.shape[0] == order.shape[0]:
            y_proba_all_arr = y_proba_all_arr[order]
        if y_score_all_arr is not None and y_score_all_arr.shape[0] == order.shape[0]:
            y_score_all_arr = y_score_all_arr[order]

    # Compute metrics via strategy (classification only)
    if eval_kind == "classification" and y_true_all_arr.size and y_pred_all_arr.size:
        metrics_result = metrics_computer.compute(
            y_true_all_arr,
            y_pred_all_arr,
            y_proba=y_proba_all_arr,
            y_score=y_score_all_arr,
        )
        confusion_payload = metrics_result.get("confusion")
        roc_raw = metrics_result.get("roc")
    else:
        confusion_payload = None
        roc_raw = None
    # Normalize confusion payload into JSON-friendly pieces
    labels_out, cm_mat, per_class, overall, confusion_macro_avg, weighted_avg = normalize_confusion(confusion_payload)

    # Normalize ROC payload into RocMetrics shape
    roc_payload = normalize_roc(roc_raw)

    # Regression diagnostics (regression only)
    regression_payload = None
    if eval_kind == "regression" and y_true_all_arr.size and y_pred_all_arr.size:
        try:
            regression_payload = build_regression_diagnostics_payload(
                y_true=y_true_all_arr,
                y_pred=y_pred_all_arr,
                seed=int(getattr(cfg.eval, "seed", 0) or 0),
            )
        except Exception:
            regression_payload = None

    n_train_avg = int(np.round(np.mean(n_train_sizes))) if n_train_sizes else 0
    n_test_avg = int(np.round(np.mean(n_test_sizes))) if n_test_sizes else 0

    if mode == "holdout":
        fold_scores_out = None
        n_splits_out = 1
        n_train_out = n_train_sizes[0] if n_train_sizes else n_train_avg
        n_test_out = n_test_sizes[0] if n_test_sizes else n_test_avg
    else:
        fold_scores_out = fold_scores
        n_splits_out = len(fold_scores)
        n_train_out = n_train_avg
        n_test_out = n_test_avg

    result: Dict[str, Any] = {
        "metric_name": cfg.eval.metric,
        "fold_scores": fold_scores_out,
        "mean_score": mean_score,
        "std_score": std_score,
        "n_splits": n_splits_out,
        "metric_value": mean_score,  # TrainResponse uses a single number; keep parity
        "confusion": {
            "labels": labels_out,
            "matrix": cm_mat,
            "per_class": per_class,
            "overall": overall,
            "macro_avg": confusion_macro_avg,
            "weighted_avg": weighted_avg,
        },
        "roc": roc_payload,
        "regression": regression_payload,
        "n_train": n_train_out,
        "n_test": n_test_out,
        "notes": [],
    }

    # --- Decoder outputs (optional) -----------------------------------------
    # For holdout, this is simply the test split.
    # For kfold CV, this is OOF: concatenated fold test splits scored by each fold model.
    if decoder_enabled:
        if eval_kind == "classification":
            try:
                result["decoder_outputs"] = build_decoder_outputs_payload(
                    decoder_cfg=decoder_cfg,
                    mode=mode,
                    y_pred_all=y_pred_all_arr,
                    y_true_all=y_true_all_arr if y_true_all_arr.size else None,
                    order=order,
                    row_indices=row_indices_arr,
                    decoder_classes=decoder_classes,
                    positive_class_label=decoder_positive_label,
                    positive_class_index=decoder_positive_index,
                    decision_scores_parts=decoder_scores_parts,
                    proba_parts=decoder_proba_parts,
                    margin_parts=decoder_margin_parts,
                    fold_ids_parts=decoder_fold_ids,
                    notes=decoder_notes_all,
                )
                for n in (decoder_notes_all or []):
                    result["notes"].append(f"Decoder outputs: {n}")
            except Exception as e:
                result["decoder_outputs"] = None
                result["notes"].append(
                    f"Decoder outputs could not be computed ({type(e).__name__}: {e})."
                )
        elif eval_kind == "regression":
            try:
                result["decoder_outputs"] = build_regression_decoder_outputs_payload(
                    decoder_cfg=decoder_cfg,
                    mode=mode,
                    y_true_all=y_true_all_arr if y_true_all_arr.size else None,
                    y_pred_all=y_pred_all_arr,
                    row_indices=row_indices_arr,
                    fold_ids_all=fold_ids_all_arr,
                    notes=[],
                )
            except Exception as e:
                result["decoder_outputs"] = None
                result["notes"].append(
                    f"Decoder outputs could not be computed ({type(e).__name__}: {e})."
                )

    # --- Feature-related notes ----------------------------------------------
    if cfg.features.method == "pca":
        result["notes"].append("Model trained on PCA-transformed features.")
    if cfg.features.method == "lda":
        result["notes"].append("LDA was fitted on the training labels.")
    if cfg.features.method == "sfs":
        result["notes"].append("SFS performed wrapper-based feature selection on training data.")

    n_features_in = int(X.shape[1]) if hasattr(X, "shape") and len(X.shape) > 1 else None
    classes_out = result["confusion"]["labels"] if result.get("confusion") and result["confusion"]["labels"] else None
    artifact_input = ArtifactBuilderInput(
        cfg=cfg,
        pipeline=effective_pipeline,
        n_train=n_train_out,
        n_test=n_test_out,
        n_features=n_features_in,
        classes=classes_out,
        kind=eval_kind,
        summary={
            "metric_name": result["metric_name"],
            "metric_value": result["metric_value"],
            "mean_score": result["mean_score"],
            "std_score": result["std_score"],
            "n_splits": result["n_splits"],
            "notes": result.get("notes", []),
        },
    )
    result["artifact"] = build_model_artifact_meta(artifact_input)
    try:
        uid = result["artifact"]["uid"]
        artifact_cache.put(uid, effective_pipeline)

        # Cache eval outputs for post-hoc export (currently used by regression UI).
        if y_pred_all_arr is not None and (y_true_all_arr is not None and y_true_all_arr.size):
            try:
                eval_outputs_cache.put(
                    uid,
                    EvalOutputs(
                        task=eval_kind,
                        indices=row_indices_arr,
                        fold_ids=fold_ids_all_arr,
                        y_true=y_true_all_arr,
                        y_pred=y_pred_all_arr,
                    ),
                )
            except Exception:
                pass
    except Exception:
        # Cache errors must not break training response.
        pass
    
    # --- Shuffle baseline ----------------------------------------------------
    n_shuffles = int(getattr(cfg.eval, "n_shuffles", 0) or 0)
    progress_id = getattr(cfg.eval, "progress_id", None)

    if n_shuffles > 0 and progress_id:
        # PRE-INIT progress so the first poll doesn't 404
        PROGRESS.init(progress_id, total=n_shuffles, label=f"Shuffling 0/{n_shuffles}…")

        baseline = make_baseline(cfg, rngm)
        # Inject progress registry + parameters for the runner
        setattr(baseline, "progress_id", progress_id)
        setattr(baseline, "_progress_total", n_shuffles)
        setattr(baseline, "_progress", PROGRESS)

        scores = np.asarray(baseline.run(X, y), dtype=float).ravel()
        ge = int(np.sum(scores >= mean_score))
        p_val = (ge + 1.0) / (scores.size + 1.0)

        result["shuffled_scores"] = [float(v) for v in scores.tolist()]
        result["p_value"] = float(p_val)
        result["notes"].append(
            f"Shuffle baseline: mean={float(np.mean(scores)):.4f} ± {float(np.std(scores)):.4f}, p≈{float(p_val):.4f}"
        )

    return result


def train_unsupervised(cfg: UnsupervisedRunConfig) -> Dict[str, Any]:
    """Train an unsupervised (clustering) model and return a JSON-safe payload.

    Notes:
      - `y` may exist in the input data but is ignored.
      - Some estimators do not support `predict` on unseen data; `apply` handling
        should be UI-gated. We still handle this defensively and emit warnings.
    """

    # --- Load training data -------------------------------------------------
    try:
        loader = make_data_loader(cfg.data)
        X, _y_ignored = loader.load()
    except Exception as e:
        raise LoadError(str(e))

    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError("Unsupervised training requires a 2D feature matrix X with at least 1 sample and 1 feature.")

    # --- RNG ----------------------------------------------------------------
    rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

    # --- Build + fit pipeline ----------------------------------------------
    pipeline = make_unsupervised_pipeline(cfg, rngm, stream="unsupervised/train")
    # sklearn clustering estimators typically accept y=None, but we call fit(X)
    pipeline.fit(X)

    # --- Extract training labels -------------------------------------------
    labels: Optional[np.ndarray] = None
    try:
        est = pipeline.steps[-1][1] if hasattr(pipeline, "steps") and pipeline.steps else pipeline
    except Exception:
        est = pipeline

    # Prefer labels_ (Agglomerative/Spectral/DBSCAN/KMeans/etc.)
    if labels is None:
        lab = getattr(est, "labels_", None)
        if lab is not None:
            labels = np.asarray(lab)

    # Mixture models (GMM/BGMM) typically use predict
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
                # We do not yet return apply outputs in the response model; this
                # is recorded into artifact extra_stats for later UI expansion.
                _ = pipeline.predict(X_apply)
            else:
                apply_notes.append(
                    f"apply requested but estimator {type(est).__name__} does not support predict(); skipping apply."  # noqa: E501
                )
        except Exception as e:
            apply_notes.append(f"apply failed: {type(e).__name__}: {e}")

    # --- Build per-sample preview table ------------------------------------
    per_sample = diag.get("per_sample") or {}
    n_rows_total = int(X.shape[0])
    preview_n = min(50, n_rows_total)

    preview_rows: list[Dict[str, Any]] = []
    # Ensure cluster_id exists
    cluster_ids = per_sample.get("cluster_id")
    if cluster_ids is None:
        cluster_ids = [int(v) for v in labels.tolist()]
        per_sample["cluster_id"] = cluster_ids

    for i in range(preview_n):
        row: Dict[str, Any] = {"index": int(i), "cluster_id": int(cluster_ids[i])}
        # attach any additional per-sample columns
        for k, v in per_sample.items():
            if k in ("cluster_id",):
                continue
            try:
                row[k] = v[i]
            except Exception:
                continue
        preview_rows.append(row)

    # --- Artifact meta + caching -------------------------------------------
    # Choose a primary metric for artifact summary (best-effort): silhouette
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

    # cache fitted pipeline
    try:
        uid = artifact_meta["uid"]
        artifact_cache.put(uid, pipeline)

        # cache per-sample outputs for export
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
        except Exception:
            pass
    except Exception:
        pass

    # --- Assemble response --------------------------------------------------
    out_metrics: Dict[str, Optional[float]] = {}
    if isinstance(metrics_dict, dict):
        for k, v in metrics_dict.items():
            out_metrics[str(k)] = safe_float_optional(v)

    warnings_all: list[str] = []
    try:
        warnings_all.extend([str(x) for x in (diag.get("warnings") or [])])
    except Exception:
        pass
    warnings_all.extend(apply_notes)

    return {
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
