from __future__ import annotations

"""Supervised training orchestration (Engine use-case).

Segment 12 introduces a single BL invocation surface under :mod:`engine.use_cases`.
This module contains supervised training orchestration that historically lived in
the backend service layer.

Correctness requirement
-----------------------
For k-fold CV, decoder outputs must be computed **out-of-fold** (OOF): each fold
model predicts on that fold's test split, and outputs are pooled. We must **not**
compute decoder outputs from a final refit model trained on all data.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine.components.evaluation.scoring import PROBA_METRICS
from engine.contracts.results.training import TrainResult
from engine.contracts.run_config import RunConfig
from engine.io.artifacts.meta import ArtifactBuilderInput, build_model_artifact_meta
from engine.runtime.caches.artifact_cache import artifact_cache
from engine.runtime.caches.eval_outputs_cache import EvalOutputs, eval_outputs_cache
from engine.use_cases._deps import resolve_seed, resolve_store
from engine.use_cases.artifacts import save_model_to_store

from engine.reporting.training.decoder_payloads import build_decoder_outputs_payload
from engine.reporting.training.metrics_payloads import normalize_confusion, normalize_roc
from engine.reporting.training.regression_payloads import (
    build_regression_decoder_outputs_payload,
    build_regression_diagnostics_payload,
)

from engine.reporting.decoder.decoder_outputs import compute_decoder_outputs

from utils.factories.baseline_factory import make_baseline
from utils.factories.data_loading_factory import make_data_loader
from utils.factories.eval_factory import make_evaluator
from utils.factories.metrics_factory import make_metrics_computer
from utils.factories.pipeline_factory import make_pipeline
from utils.factories.sanity_factory import make_sanity_checker
from utils.factories.split_factory import make_splitter
from utils.permutations.rng import RngManager
from sklearn.pipeline import Pipeline


def _unpack_split(split: Any) -> Tuple[Any, Any, Any, Any, Optional[np.ndarray]]:
    """Normalize splitter outputs.

    Supported formats:
      - (Xtr, Xte, ytr, yte)
      - (idx_tr, idx_te, Xtr, Xte, ytr, yte)
      - (Xtr, Xte, ytr, yte, idx_tr, idx_te)
    """

    if isinstance(split, (tuple, list)) and len(split) == 4:
        Xtr, Xte, ytr, yte = split
        return Xtr, Xte, ytr, yte, None

    if isinstance(split, (tuple, list)) and len(split) == 6:
        a0, a1, a2, a3, a4, a5 = split

        def _looks_like_indices(x: Any) -> bool:
            try:
                arr = np.asarray(x)
                return arr.ndim == 1 and arr.size >= 1 and np.issubdtype(arr.dtype, np.integer)
            except Exception:
                return False

        # indices-first
        if _looks_like_indices(a0) and _looks_like_indices(a1):
            Xtr, Xte, ytr, yte = a2, a3, a4, a5
            return Xtr, Xte, ytr, yte, np.asarray(a1, dtype=int).ravel()

        # indices-last
        if _looks_like_indices(a4) and _looks_like_indices(a5):
            Xtr, Xte, ytr, yte = a0, a1, a2, a3
            return Xtr, Xte, ytr, yte, np.asarray(a5, dtype=int).ravel()

        # fallback: assume indices-last shape
        Xtr, Xte, ytr, yte = a0, a1, a2, a3
        return Xtr, Xte, ytr, yte, None

    raise ValueError(
        "Splitter yielded an unsupported split tuple; expected 4 or 6 items. "
        f"Got {type(split).__name__} of length {len(split) if isinstance(split, (tuple, list)) else 'n/a'}."
    )


def train_supervised(
    run_config: RunConfig,
    *,
    store: Any = None,
    rng: Optional[int] = None,
) -> TrainResult:
    """Train a supervised model and return a :class:`~engine.contracts.results.training.TrainResult`."""

    cfg = run_config

    # --- Load data ---------------------------------------------------------
    loader = make_data_loader(cfg.data)
    X, y = loader.load()

    # --- Checks ------------------------------------------------------------
    make_sanity_checker().check(X, y)

    # --- Dependencies ------------------------------------------------------
    store_resolved = resolve_store(store)
    seed = resolve_seed(int(rng) if rng is not None else getattr(cfg.eval, "seed", None), fallback=0)
    rngm = RngManager(seed)

    # --- Split -------------------------------------------------------------
    split_seed = rngm.child_seed("train/split")
    mode = getattr(cfg.split, "mode")
    if mode not in ("holdout", "kfold"):
        raise ValueError(f"Unsupported split mode: {mode!r}")
    splitter = make_splitter(cfg.split, seed=split_seed)

    # --- Task kind + evaluators -------------------------------------------
    from shared_schemas.model_configs import get_model_task

    eval_kind = get_model_task(cfg.model)
    metrics_computer = make_metrics_computer(kind=eval_kind)

    # Regression must not stratify (defensive)
    if eval_kind == "regression" and hasattr(cfg.split, "stratified"):
        cfg.split.stratified = False

    evaluator = make_evaluator(cfg.eval, kind=eval_kind)

    # --- Decoder outputs toggles ------------------------------------------
    decoder_cfg = getattr(cfg.eval, "decoder", None)
    decoder_enabled = bool(getattr(decoder_cfg, "enabled", False)) if decoder_cfg is not None else False
    decoder_positive_label = getattr(decoder_cfg, "positive_class_label", None) if decoder_cfg is not None else None
    decoder_include_scores = bool(getattr(decoder_cfg, "include_decision_scores", True)) if decoder_cfg is not None else True
    decoder_include_probabilities = bool(getattr(decoder_cfg, "include_probabilities", True)) if decoder_cfg is not None else True
    decoder_calibrate_probabilities = bool(getattr(decoder_cfg, "calibrate_probabilities", False)) if decoder_cfg is not None else False

    decoder_scores_parts: List[Optional[np.ndarray]] = []
    decoder_proba_parts: List[Optional[np.ndarray]] = []
    decoder_margin_parts: List[Optional[np.ndarray]] = []
    decoder_notes_all: List[str] = []
    decoder_classes: Optional[np.ndarray] = None
    decoder_positive_index: Optional[int] = None
    decoder_fold_ids: List[np.ndarray] = []

    # --- Fit / evaluate over splits ---------------------------------------
    fold_scores: List[float] = []
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_proba_all: List[np.ndarray] = []
    y_score_all: List[np.ndarray] = []
    test_indices_parts: List[Optional[np.ndarray]] = []
    eval_fold_ids_parts: List[np.ndarray] = []
    n_train_sizes: List[int] = []
    n_test_sizes: List[int] = []

    last_pipeline: Optional[Pipeline] = None

    for fold_id, split in enumerate(splitter.split(X, y), start=1):
        Xtr, Xte, ytr, yte, idx_te = _unpack_split(split)

        pipeline = make_pipeline(cfg, rngm, stream=f"{mode}/fold{fold_id}")
        pipeline.fit(Xtr, ytr)
        y_pred = pipeline.predict(Xte)
        last_pipeline = pipeline

        # indices for re-ordering if available
        if idx_te is not None:
            test_indices_parts.append(np.asarray(idx_te, dtype=int).ravel())
        else:
            test_indices_parts.append(None)

        # fold ids per row
        try:
            n_fold_rows = int(np.asarray(y_pred).shape[0])
        except Exception:
            n_fold_rows = int(np.asarray(Xte).shape[0])
        eval_fold_ids_parts.append(np.full((n_fold_rows,), fold_id, dtype=int))

        metric_name = cfg.eval.metric
        y_proba = None
        y_score = None

        if eval_kind == "classification":
            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(Xte)
            if hasattr(pipeline, "decision_function"):
                y_score = pipeline.decision_function(Xte)
            if metric_name in PROBA_METRICS and y_proba is None and y_score is None:
                raise ValueError(
                    f"Metric '{metric_name}' requires predict_proba or decision_function, "
                    f"but estimator {type(pipeline).__name__} has neither."
                )

        score_val = evaluator.score(yte, y_pred=y_pred, y_proba=y_proba, y_score=y_score)
        fold_scores.append(float(score_val))

        y_true_all.append(np.asarray(yte))
        y_pred_all.append(np.asarray(y_pred))
        if y_proba is not None:
            y_proba_all.append(np.asarray(y_proba))
        if y_score is not None:
            y_score_all.append(np.asarray(y_score))

        if decoder_enabled and eval_kind == "classification":
            decoder_fold_ids.append(np.full((n_fold_rows,), fold_id, dtype=int))
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
                decoder_proba_parts.append(np.asarray(dec.proba) if dec.proba is not None else None)
                decoder_margin_parts.append(np.asarray(dec.margin) if dec.margin is not None else None)
                if dec.notes:
                    decoder_notes_all.extend([str(x) for x in dec.notes])
            except Exception as e:
                decoder_scores_parts.append(None)
                decoder_proba_parts.append(None)
                decoder_margin_parts.append(None)
                decoder_notes_all.append(
                    f"decoder outputs failed on fold {fold_id}: {type(e).__name__}: {e}"
                )

        n_train_sizes.append(int(np.asarray(Xtr).shape[0]))
        n_test_sizes.append(int(np.asarray(Xte).shape[0]))

    if last_pipeline is None:
        raise ValueError("No folds produced by splitter; cannot train model.")

    refit_pipeline = None
    if mode == "kfold":
        refit_pipeline = make_pipeline(cfg, rngm, stream="kfold/refit")
        refit_pipeline.fit(X, y)
    effective_pipeline = refit_pipeline if refit_pipeline is not None else last_pipeline

    # --- Aggregate metrics -------------------------------------------------
    if not fold_scores:
        fold_scores = [float("nan")]
    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    y_true_all_arr = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred_all_arr = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    fold_ids_all_arr = np.concatenate(eval_fold_ids_parts) if eval_fold_ids_parts else None

    order = None
    row_indices_arr: Optional[np.ndarray] = None
    if test_indices_parts and all(p is not None for p in test_indices_parts):
        idx_all_arr = np.concatenate([np.asarray(p) for p in test_indices_parts], axis=0)
        if idx_all_arr.shape[0] == y_pred_all_arr.shape[0]:
            order = np.argsort(idx_all_arr, kind="stable")
            row_indices_arr = idx_all_arr[order]
            y_true_all_arr = y_true_all_arr[order]
            y_pred_all_arr = y_pred_all_arr[order]
            if fold_ids_all_arr is not None and fold_ids_all_arr.shape[0] == order.shape[0]:
                fold_ids_all_arr = fold_ids_all_arr[order]
    if row_indices_arr is None:
        row_indices_arr = np.arange(int(np.asarray(y_pred_all_arr).shape[0]), dtype=int)

    y_proba_all_arr = np.concatenate(y_proba_all, axis=0) if y_proba_all else None
    y_score_all_arr = np.concatenate(y_score_all, axis=0) if y_score_all else None
    if order is not None:
        if y_proba_all_arr is not None and y_proba_all_arr.shape[0] == order.shape[0]:
            y_proba_all_arr = y_proba_all_arr[order]
        if y_score_all_arr is not None and y_score_all_arr.shape[0] == order.shape[0]:
            y_score_all_arr = y_score_all_arr[order]

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

    labels_out, cm_mat, per_class, overall, confusion_macro_avg, weighted_avg = normalize_confusion(confusion_payload)
    roc_payload = normalize_roc(roc_raw)

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
        "metric_value": mean_score,
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

    # --- Decoder outputs ---------------------------------------------------
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

    # --- Feature-related notes --------------------------------------------
    if cfg.features.method == "pca":
        result["notes"].append("Model trained on PCA-transformed features.")
    if cfg.features.method == "lda":
        result["notes"].append("LDA was fitted on the training labels.")
    if cfg.features.method == "sfs":
        result["notes"].append(
            "SFS performed wrapper-based feature selection on training data."
        )

    # --- Artifact meta + caching + persistence ----------------------------
    n_features_in = int(np.asarray(X).shape[1]) if hasattr(X, "shape") and len(np.asarray(X).shape) > 1 else None
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
    artifact_meta = build_model_artifact_meta(artifact_input)
    result["artifact"] = artifact_meta

    # Persist
    try:
        save_model_to_store(store_resolved, effective_pipeline, artifact_meta)
    except Exception:
        # persistence is best-effort at this stage
        pass

    # Cache fitted pipeline (process-local)
    try:
        uid = artifact_meta["uid"]
        artifact_cache.put(uid, effective_pipeline)
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
        pass

    # --- Shuffle baseline --------------------------------------------------
    n_shuffles = int(getattr(cfg.eval, "n_shuffles", 0) or 0)
    if n_shuffles > 0:
        baseline = make_baseline(cfg, rngm)
        # Baseline runner expects these private fields (legacy)
        try:
            baseline._progress_total = n_shuffles  # type: ignore[attr-defined]
            baseline._progress_done = 0  # type: ignore[attr-defined]
            baseline._progress_id = None  # type: ignore[attr-defined]
        except Exception:
            pass
        baseline_out = baseline.run(X, y)
        try:
            result["shuffle_baseline"] = {
                "n_shuffles": n_shuffles,
                "scores": baseline_out.scores,
                "p_value": baseline_out.p_value,
            }
        except Exception:
            pass

    return TrainResult.model_validate(result)
