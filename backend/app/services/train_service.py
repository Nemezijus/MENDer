import numpy as np
from typing import Dict, Any

from shared_schemas.run_config import RunConfig
from shared_schemas.model_configs import get_model_task
from utils.factories.data_loading_factory import make_data_loader
from utils.factories.sanity_factory import make_sanity_checker
from utils.factories.split_factory import make_splitter
from utils.factories.pipeline_factory import make_pipeline
from utils.factories.eval_factory import make_evaluator
from utils.factories.metrics_factory import make_metrics_computer
from utils.permutations.rng import RngManager
from utils.factories.baseline_factory import make_baseline
from utils.persistence.modelArtifact import ArtifactBuilderInput, build_model_artifact_meta
from utils.persistence.artifact_cache import artifact_cache
from utils.postprocessing.scoring import PROBA_METRICS
from utils.postprocessing.decoder_outputs import compute_decoder_outputs

from ..adapters.io_adapter import LoadError
from ..progress.registry import PROGRESS

from .common.json_safety import safe_float_list, safe_float_scalar
from .training.metrics_payloads import normalize_confusion, normalize_roc
from .training.decoder_payloads import build_decoder_outputs_payload




def train(cfg: RunConfig) -> Dict[str, Any]:
    # --- Load data -----------------------------------------------------------
    try:
        loader = make_data_loader(cfg.data)
        X, y = loader.load()
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
            # Always track fold ID per test row for UI/diagnostics.
            n_fold_rows = int(np.asarray(y_pred).shape[0])
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
        "n_train": n_train_out,
        "n_test": n_test_out,
        "notes": [],
    }

    # --- Decoder outputs (optional) -----------------------------------------
    # For holdout, this is simply the test split.
    # For kfold CV, this is OOF: concatenated fold test splits scored by each fold model.
    if decoder_enabled and eval_kind == "classification":
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
