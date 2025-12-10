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

from ..adapters.io_adapter import LoadError
from ..progress.registry import PROGRESS


def _safe_float_list(arr) -> list[float]:
    """Convert array-like to a JSON-safe list of finite floats."""
    a = np.asarray(arr, dtype=float)
    # Replace NaN with 0.0, +inf with 1.0, -inf with 0.0 (arbitrary but finite).
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    return a.tolist()


def _safe_float_scalar(x: float) -> float:
    """Make a single float JSON-safe (no NaN/inf)."""
    return float(np.nan_to_num(float(x), nan=0.0, posinf=1.0, neginf=0.0))


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
    n_train_sizes, n_test_sizes = [], []

    for fold_id, (Xtr, Xte, ytr, yte) in enumerate(splitter.split(X, y), start=1):
        pipeline = make_pipeline(cfg, rngm, stream=f"{mode}/fold{fold_id}")
        pipeline.fit(Xtr, ytr)
        y_pred = pipeline.predict(Xte)
        last_pipeline = pipeline

        metric_name = cfg.eval.metric
        y_proba = None
        y_score = None

        # For classification, we try to compute scores/probabilities once per fold.
        if eval_kind == "classification":
            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(Xte)
            elif hasattr(pipeline, "decision_function"):
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

    y_proba_all_arr = np.concatenate(y_proba_all, axis=0) if y_proba_all else None
    y_score_all_arr = np.concatenate(y_score_all, axis=0) if y_score_all else None

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
    if confusion_payload is not None:
        labels_arr = confusion_payload["labels"]
        matrix_arr = confusion_payload["matrix"]

        # Convert labels to proper JSON-friendly types
        if hasattr(labels_arr, "tolist"):
            labels_base = labels_arr.tolist()
        else:
            labels_base = list(labels_arr)

        labels_out = [
            str(l) if not isinstance(l, (int, float)) else l for l in labels_base
        ]

        if hasattr(matrix_arr, "tolist"):
            cm_mat = matrix_arr.tolist()
        else:
            cm_mat = matrix_arr

        per_class = confusion_payload.get("per_class") or []
        overall = confusion_payload.get("global") or None
        confusion_macro_avg = confusion_payload.get("macro_avg") or None
        weighted_avg = confusion_payload.get("weighted_avg") or None
    else:
        labels_out = []
        cm_mat = []
        per_class = []
        overall = None
        confusion_macro_avg = None
        weighted_avg = None

    # Normalize ROC payload into RocMetrics shape
    roc_payload = None
    if roc_raw is not None:
        if "pos_label" in roc_raw:
            # Binary ROC
            auc_val = _safe_float_scalar(roc_raw["auc"])
            curve = {
                "label": roc_raw["pos_label"],
                "fpr": _safe_float_list(roc_raw["fpr"]),
                "tpr": _safe_float_list(roc_raw["tpr"]),
                "thresholds": _safe_float_list(roc_raw["thresholds"]),
                "auc": auc_val,
            }
            roc_payload = {
                "kind": "binary",
                "curves": [curve],
                "labels": None,
                "macro_auc": auc_val,
                "positive_label": roc_raw["pos_label"],
            }
        elif "per_class" in roc_raw:
            # Multiclass one-vs-rest ROC
            labels_arr = roc_raw.get("labels")
            if hasattr(labels_arr, "tolist"):
                labels_list = labels_arr.tolist()
            else:
                labels_list = list(labels_arr) if labels_arr is not None else None

            curves = []
            for entry in roc_raw["per_class"]:
                curves.append(
                    {
                        "label": entry["label"],
                        "fpr": _safe_float_list(entry["fpr"]),
                        "tpr": _safe_float_list(entry["tpr"]),
                        "thresholds": _safe_float_list(entry["thresholds"]),
                        "auc": _safe_float_scalar(entry["auc"]),
                    }
                )

            roc_macro_avg = roc_raw.get("macro_avg") or {}
            macro_auc = (
                _safe_float_scalar(roc_macro_avg["auc"])
                if "auc" in roc_macro_avg
                else None
            )

            # Optionally add a macro-average ROC curve if fpr/tpr are available
            macro_fpr = roc_macro_avg.get("fpr")
            macro_tpr = roc_macro_avg.get("tpr")
            if macro_fpr is not None and macro_tpr is not None:
                curves.append(
                    {
                        "label": "macro",
                        "fpr": _safe_float_list(macro_fpr),
                        "tpr": _safe_float_list(macro_tpr),
                        # thresholds for macro-average are not well-defined;
                        # use empty list to keep the schema consistent.
                        "thresholds": [],
                        "auc": macro_auc if macro_auc is not None else 0.0,
                    }
                )

            # Optionally add a micro-average ROC curve if present
            roc_micro_avg = roc_raw.get("micro_avg") or {}
            micro_auc: float | None = None
            micro_fpr = roc_micro_avg.get("fpr")
            micro_tpr = roc_micro_avg.get("tpr")
            micro_thresholds = roc_micro_avg.get("thresholds")

            if (
                micro_fpr is not None
                and micro_tpr is not None
                and micro_thresholds is not None
            ):
                micro_auc = _safe_float_scalar(roc_micro_avg.get("auc", float("nan")))
                curves.append(
                    {
                        "label": "micro",
                        "fpr": _safe_float_list(micro_fpr),
                        "tpr": _safe_float_list(micro_tpr),
                        "thresholds": _safe_float_list(micro_thresholds),
                        "auc": micro_auc,
                    }
                )

            roc_payload = {
                "kind": "multiclass",
                "curves": curves,
                "labels": labels_list,
                "macro_auc": macro_auc,
            }
            if micro_auc is not None:
                roc_payload["micro_auc"] = micro_auc

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
