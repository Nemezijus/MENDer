import numpy as np

from engine.contracts.results.decoder import DecoderOutputs
from typing import Dict, Any, Optional, Sequence

from shared_schemas.ensemble_run_config import EnsembleRunConfig

from utils.factories.data_loading_factory import make_data_loader
from utils.factories.sanity_factory import make_sanity_checker
from utils.factories.split_factory import make_splitter
from utils.factories.eval_factory import make_evaluator
from utils.factories.metrics_factory import make_metrics_computer
from utils.factories.ensemble_factory import make_ensemble_strategy
from utils.permutations.rng import RngManager
from engine.io.artifacts.meta import ArtifactBuilderInput, build_model_artifact_meta
from engine.runtime.caches.artifact_cache import artifact_cache
from engine.runtime.caches.eval_outputs_cache import EvalOutputs, eval_outputs_cache
from engine.components.evaluation.scoring import PROBA_METRICS

from ..adapters.io_adapter import LoadError

from sklearn.ensemble import VotingClassifier, VotingRegressor
from shared_schemas.ensemble_configs import (
    VotingEnsembleConfig,
    BaggingEnsembleConfig,
    AdaBoostEnsembleConfig,
    XGBoostEnsembleConfig,
)

from engine.reporting.ensembles.voting_ensemble_reporting import (
    VotingEnsembleReportAccumulator,
    VotingEnsembleRegressorReportAccumulator,
)
from engine.reporting.ensembles.bagging_ensemble_reporting import (
    BaggingEnsembleReportAccumulator,
    BaggingEnsembleRegressorReportAccumulator,
)
from engine.reporting.ensembles.adaboost_ensemble_reporting import (
    AdaBoostEnsembleReportAccumulator,
    AdaBoostEnsembleRegressorReportAccumulator,
)
from engine.reporting.ensembles.xgboost_ensemble_reporting import XGBoostEnsembleReportAccumulator

from engine.components.prediction.decoder_extraction import compute_decoder_outputs_raw
from engine.components.prediction.decoder_api import build_decoder_outputs_from_arrays

from .training.regression_payloads import (
    build_regression_diagnostics_payload,
    build_regression_decoder_outputs_payload,
)

from .common.json_safety import safe_float_list, safe_float_scalar, dedupe_preserve_order
from .ensembles.helpers import (
    _unwrap_final_estimator,
    _transform_through_pipeline,
    _slice_X_by_features,
    _get_classes_arr,
    _should_decode_from_index_space,
    _encode_y_true_to_index,
    _friendly_ensemble_training_error,
    _extract_base_estimator_algo_from_cfg,
)

from .ensembles.reports.voting import update_voting_report
from .ensembles.reports.bagging import update_bagging_report
from .ensembles.reports.adaboost import update_adaboost_report
from .ensembles.reports.xgboost import update_xgboost_report

def train_ensemble(cfg: EnsembleRunConfig) -> Dict[str, Any]:
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

    # --- Decoder outputs (per-sample decision scores / probabilities) -------
    decoder_cfg = getattr(cfg.eval, "decoder", None)
    decoder_enabled = bool(getattr(decoder_cfg, "enabled", False)) if decoder_cfg is not None else False
    decoder_positive_label = getattr(decoder_cfg, "positive_class_label", None) if decoder_cfg is not None else None

    # Optional toggles (default True if missing)
    decoder_include_scores = bool(getattr(decoder_cfg, "include_decision_scores", True)) if decoder_cfg is not None else True
    decoder_include_probabilities = bool(getattr(decoder_cfg, "include_probabilities", True)) if decoder_cfg is not None else True
    decoder_include_margin = bool(getattr(decoder_cfg, "include_margin", True)) if decoder_cfg is not None else True
    decoder_calibrate_probabilities = bool(getattr(decoder_cfg, "calibrate_probabilities", False)) if decoder_cfg is not None else False

    # Preview row cap for UI payload (full export is handled separately)
    decoder_max_preview_rows = int(getattr(decoder_cfg, "max_preview_rows", 200) or 200) if decoder_cfg is not None else 0

    # Accumulators
    decoder_scores_all: list[np.ndarray] = []
    decoder_proba_all: list[np.ndarray] = []
    decoder_proba_source: Optional[str] = None
    decoder_margin_all: list[np.ndarray] = []
    decoder_notes: list[str] = []
    decoder_classes: Optional[np.ndarray] = None
    decoder_positive_index: Optional[int] = None
    decoder_fold_ids: list[np.ndarray] = []

    # --- Split (hold-out or k-fold) -----------------------------------------
    split_seed = rngm.child_seed("ensemble/train/split")

    try:
        mode = getattr(cfg.split, "mode")
    except AttributeError:
        raise ValueError("EnsembleRunConfig.split is missing required 'mode' attribute")

    if mode not in ("holdout", "kfold"):
        raise ValueError(f"Unsupported split mode in ensemble train service: {mode!r}")

    # Decide evaluation kind based on ensemble config (authoritative)
    ensemble_strategy = make_ensemble_strategy(cfg)
    eval_kind = ensemble_strategy.expected_kind()

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
    # Optional original row indices for evaluation rows (when splitter yields them)
    test_indices_parts: list[np.ndarray | None] = []
    # Fold id per evaluation row (useful for CV export)
    eval_fold_ids_parts: list[np.ndarray] = []
    n_train_sizes, n_test_sizes = [], []

    last_model = None

    # --- Accumulator setup (for ensemble reports) ------------------------
    voting_cls_acc: VotingEnsembleReportAccumulator | None = None
    voting_reg_acc: VotingEnsembleRegressorReportAccumulator | None = None
    is_voting_report = isinstance(cfg.ensemble, VotingEnsembleConfig)

    bagging_cls_acc: BaggingEnsembleReportAccumulator | None = None
    bagging_reg_acc: BaggingEnsembleRegressorReportAccumulator | None = None
    is_bagging_report = isinstance(cfg.ensemble, BaggingEnsembleConfig)

    adaboost_cls_acc: AdaBoostEnsembleReportAccumulator | None = None
    adaboost_reg_acc: AdaBoostEnsembleRegressorReportAccumulator | None = None
    is_adaboost_report = isinstance(cfg.ensemble, AdaBoostEnsembleConfig)

    xgb_acc: XGBoostEnsembleReportAccumulator | None = None
    # XGBoost report is task-agnostic (works for classification + regression)
    is_xgboost_report = isinstance(cfg.ensemble, XGBoostEnsembleConfig)

    for fold_id, split in enumerate(splitter.split(X, y), start=1):
        # Split can be 4-tuple (Xtr,Xte,ytr,yte) or 6-tuple (..., idx_tr, idx_te)
        if isinstance(split, (tuple, list)) and len(split) == 4:
            Xtr, Xte, ytr, yte = split
            idx_te = None
        elif isinstance(split, (tuple, list)) and len(split) == 6:
            Xtr, Xte, ytr, yte, _idx_tr, idx_te = split
        else:
            raise ValueError(
                "Splitter yielded an unsupported split tuple; expected 4 or 6 items. "
                f"Got {type(split).__name__} of length {len(split) if isinstance(split, (tuple, list)) else 'n/a'}."
            )
        try:
            model = ensemble_strategy.fit(
                Xtr,
                ytr,
                rngm=rngm,
                stream=f"{mode}/fold{fold_id}",
            )
            last_model = model

            y_pred = model.predict(Xte)

            # Keep test indices for optional re-ordering back to original row order.
            if idx_te is not None:
                test_indices_parts.append(np.asarray(idx_te, dtype=int).ravel())
            else:
                test_indices_parts.append(None)

            # Always keep fold id per evaluation row (works for both holdout and k-fold).
            try:
                n_fold_rows = int(np.asarray(y_pred).shape[0])
            except Exception:
                n_fold_rows = int(Xte.shape[0])
            eval_fold_ids_parts.append(np.full((n_fold_rows,), fold_id, dtype=int))

            # --- Optional decoder outputs (classification only) ----------------
            if decoder_enabled and eval_kind == "classification":
                # Always track fold ID per test row for UI/diagnostics.
                n_fold_rows = int(np.asarray(y_pred).shape[0])
                decoder_fold_ids.append(np.full((n_fold_rows,), fold_id, dtype=int))

                try:
                    dec = compute_decoder_outputs_raw(
                        model,
                        Xte,
                        positive_class_label=decoder_positive_label,
                        include_decision_scores=decoder_include_scores,
                        include_probabilities=decoder_include_probabilities,
                        calibrate_probabilities=decoder_calibrate_probabilities,
                    )
                    if decoder_classes is None and dec.classes is not None:
                        decoder_classes = np.asarray(dec.classes)
                    # Store positive index if available
                    if decoder_positive_index is None and dec.positive_class_index is not None:
                        decoder_positive_index = int(dec.positive_class_index)

                    if dec.decision_scores is not None:
                        decoder_scores_all.append(np.asarray(dec.decision_scores))
                    if dec.proba is not None:
                        decoder_proba_all.append(np.asarray(dec.proba))
                    # Track probability source across folds
                    try:
                        src = getattr(dec, 'proba_source', None)
                        if src is not None:
                            if decoder_proba_source is None:
                                decoder_proba_source = str(src)
                            elif str(src) != str(decoder_proba_source):
                                decoder_proba_source = 'mixed'
                    except Exception:
                        pass
                    if decoder_include_margin and dec.margin is not None:
                        decoder_margin_all.append(np.asarray(dec.margin))
                    if dec.notes:
                        decoder_notes.extend([str(x) for x in dec.notes])
                except Exception as e:
                    decoder_notes.append(
                        f"decoder outputs failed on fold {fold_id}: {type(e).__name__}: {e}"
                    )
        except Exception as e:
            raise _friendly_ensemble_training_error(e, cfg, fold_id=fold_id) from e
        # ---------------- Voting report (classification + regression) ----------------
        if is_voting_report:
            voting_cls_acc, voting_reg_acc = update_voting_report(
                cfg=cfg,
                eval_kind=eval_kind,
                model=model,
                Xtr=Xtr,
                Xte=Xte,
                ytr=ytr,
                yte=yte,
                y_pred=y_pred,
                fold_id=fold_id,
                evaluator=evaluator,
                voting_cls_acc=voting_cls_acc,
                voting_reg_acc=voting_reg_acc,
            )

# ---------------- Bagging report ----------------
        if is_bagging_report:
            bagging_cls_acc, bagging_reg_acc = update_bagging_report(
                cfg=cfg,
                eval_kind=eval_kind,
                model=model,
                Xte=Xte,
                yte=yte,
                y_pred=y_pred,
                fold_id=fold_id,
                evaluator=evaluator,
                bagging_cls_acc=bagging_cls_acc,
                bagging_reg_acc=bagging_reg_acc,
            )
        # ---------------- AdaBoost report (classification + regression) ----------------
        if is_adaboost_report:
            adaboost_cls_acc, adaboost_reg_acc = update_adaboost_report(
                cfg=cfg,
                eval_kind=eval_kind,
                model=model,
                Xte=Xte,
                yte=yte,
                y_pred=y_pred,
                fold_id=fold_id,
                evaluator=evaluator,
                adaboost_cls_acc=adaboost_cls_acc,
                adaboost_reg_acc=adaboost_reg_acc,
            )
        # ---------------- XGBoost report (adjusted) ----------------
        if is_xgboost_report:
            xgb_acc = update_xgboost_report(
                cfg=cfg,
                model=model,
                Xtr=Xtr,
                Xte=Xte,
                ytr=ytr,
                yte=yte,
                y_pred=y_pred,
                fold_id=fold_id,
                xgb_acc=xgb_acc,
            )


        metric_name = cfg.eval.metric
        y_proba = None
        y_score = None

        if eval_kind == "classification":
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(Xte)
                elif hasattr(model, "decision_function"):
                    y_score = model.decision_function(Xte)
            except Exception as e:
                raise _friendly_ensemble_training_error(e, cfg, fold_id=fold_id) from e

            if metric_name in PROBA_METRICS and y_proba is None and y_score is None:
                raise ValueError(
                    f"Metric '{metric_name}' requires predict_proba or decision_function, "
                    f"but estimator {type(model).__name__} has neither."
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

    if last_model is None:
        raise ValueError("No folds produced by splitter; cannot train ensemble.")

    refit_model = None
    if mode == "kfold":
        try:
            refit_model = ensemble_strategy.fit(
                X,
                y,
                rngm=rngm,
                stream="kfold/refit",
            )
        except Exception as e:
            raise _friendly_ensemble_training_error(e, cfg, fold_id=None) from e

    effective_model = refit_model if refit_model is not None else last_model

    # --- Aggregate scores and metrics (confusion + ROC) ---------------------
    if not fold_scores:
        fold_scores = [float("nan")]

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    y_true_all_arr = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred_all_arr = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    fold_ids_all_arr = np.concatenate(eval_fold_ids_parts, axis=0) if eval_fold_ids_parts else None

    y_proba_all_arr = np.concatenate(y_proba_all, axis=0) if y_proba_all else None
    y_score_all_arr = np.concatenate(y_score_all, axis=0) if y_score_all else None

    # If splitter provided original indices, reorder pooled evaluation arrays back
    # to the original row order.
    order = None
    row_indices_arr: np.ndarray | None = None
    if test_indices_parts and all(p is not None for p in test_indices_parts):
        try:
            idx_all_arr = np.concatenate([np.asarray(p) for p in test_indices_parts], axis=0)
            order = np.argsort(idx_all_arr)
            row_indices_arr = idx_all_arr[order]
            y_true_all_arr = y_true_all_arr[order]
            y_pred_all_arr = y_pred_all_arr[order]
            if fold_ids_all_arr is not None and fold_ids_all_arr.shape[0] == order.shape[0]:
                fold_ids_all_arr = fold_ids_all_arr[order]
            if y_proba_all_arr is not None and y_proba_all_arr.shape[0] == order.shape[0]:
                y_proba_all_arr = y_proba_all_arr[order]
            if y_score_all_arr is not None and y_score_all_arr.shape[0] == order.shape[0]:
                y_score_all_arr = y_score_all_arr[order]
        except Exception:
            order = None
            row_indices_arr = None

    if row_indices_arr is None:
        # Fall back to consecutive indices in pooled eval order.
        row_indices_arr = np.arange(int(y_pred_all_arr.shape[0]), dtype=int)

    # --- Aggregate decoder outputs across folds (out-of-sample) -------------
    decoder_payload: Optional[DecoderOutputs] = None
    if decoder_enabled and eval_kind == "classification":
        try:
            ds_arr = np.concatenate(decoder_scores_all, axis=0) if decoder_scores_all else None
            pr_arr = np.concatenate(decoder_proba_all, axis=0) if decoder_proba_all else None
            mg_arr = np.concatenate(decoder_margin_all, axis=0) if decoder_margin_all else None
            fold_ids_arr = np.concatenate(decoder_fold_ids, axis=0) if decoder_fold_ids else None

            # If we reordered evaluation rows, reorder decoder arrays to match.
            if order is not None:
                if ds_arr is not None and ds_arr.shape[0] == order.shape[0]:
                    ds_arr = ds_arr[order]
                if pr_arr is not None and pr_arr.shape[0] == order.shape[0]:
                    pr_arr = pr_arr[order]
                if mg_arr is not None and mg_arr.shape[0] == order.shape[0]:
                    mg_arr = mg_arr[order]
                if fold_ids_arr is not None and fold_ids_arr.shape[0] == order.shape[0]:
                    fold_ids_arr = fold_ids_arr[order]

            decoder_payload = build_decoder_outputs_from_arrays(
                y_pred=y_pred_all_arr,
                y_true=(
                    y_true_all_arr
                    if (y_true_all_arr is not None and np.asarray(y_true_all_arr).size)
                    else None
                ),
                indices=row_indices_arr,
                classes=decoder_classes,
                positive_class_label=decoder_positive_label,
                positive_class_index=decoder_positive_index,
                decision_scores=ds_arr,
                proba=pr_arr,
                proba_source=decoder_proba_source,
                margin=mg_arr,
                fold_ids=fold_ids_arr,
                notes=dedupe_preserve_order(decoder_notes),
                max_preview_rows=(
                    decoder_max_preview_rows if decoder_max_preview_rows > 0 else None
                ),
                include_summary=True,
            )
        except Exception as e:
            decoder_payload = DecoderOutputs.model_validate(
                {
                    "classes": decoder_classes.tolist() if decoder_classes is not None else None,
                    "positive_class_label": decoder_positive_label,
                    "positive_class_index": decoder_positive_index,
                    "has_decision_scores": False,
                    "has_proba": False,
                    "notes": dedupe_preserve_order(
                        decoder_notes + [f"decoder aggregation failed: {type(e).__name__}: {e}"]
                    ),
                    "n_rows_total": int(len(y_pred_all_arr)) if y_pred_all_arr is not None else None,
                    "preview_rows": [],
                }
            )

    # Regression decoder outputs (no decision scores / probabilities)
    if decoder_enabled and eval_kind == "regression":
        try:
            decoder_payload = build_regression_decoder_outputs_payload(
                decoder_cfg=decoder_cfg,
                mode=mode,
                y_true_all=y_true_all_arr if y_true_all_arr.size else None,
                y_pred_all=y_pred_all_arr,
                row_indices=row_indices_arr,
                fold_ids_all=fold_ids_all_arr,
                notes=[],
            )
        except Exception:
            decoder_payload = None

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

    # Normalize confusion payload
    if confusion_payload is not None:
        labels_arr = confusion_payload["labels"]
        matrix_arr = confusion_payload["matrix"]

        labels_base = labels_arr.tolist() if hasattr(labels_arr, "tolist") else list(labels_arr)
        labels_out = [str(l) if not isinstance(l, (int, float)) else l for l in labels_base]

        cm_mat = matrix_arr.tolist() if hasattr(matrix_arr, "tolist") else matrix_arr

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

    # Normalize ROC payload
    roc_payload = None
    if roc_raw is not None:
        if "pos_label" in roc_raw:
            auc_val = safe_float_scalar(roc_raw["auc"])
            curve = {
                "label": roc_raw["pos_label"],
                "fpr": safe_float_list(roc_raw["fpr"]),
                "tpr": safe_float_list(roc_raw["tpr"]),
                "thresholds": safe_float_list(roc_raw["thresholds"]),
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
            labels_arr = roc_raw.get("labels")
            labels_list = (
                labels_arr.tolist()
                if hasattr(labels_arr, "tolist")
                else (list(labels_arr) if labels_arr is not None else None)
            )

            curves = []
            for entry in roc_raw["per_class"]:
                curves.append(
                    {
                        "label": entry["label"],
                        "fpr": safe_float_list(entry["fpr"]),
                        "tpr": safe_float_list(entry["tpr"]),
                        "thresholds": safe_float_list(entry["thresholds"]),
                        "auc": safe_float_scalar(entry["auc"]),
                    }
                )

            roc_macro_avg = roc_raw.get("macro_avg") or {}
            macro_auc = safe_float_scalar(roc_macro_avg["auc"]) if "auc" in roc_macro_avg else None

            macro_fpr = roc_macro_avg.get("fpr")
            macro_tpr = roc_macro_avg.get("tpr")
            if macro_fpr is not None and macro_tpr is not None:
                curves.append(
                    {
                        "label": "macro",
                        "fpr": safe_float_list(macro_fpr),
                        "tpr": safe_float_list(macro_tpr),
                        "thresholds": [],
                        "auc": macro_auc if macro_auc is not None else 0.0,
                    }
                )

            roc_micro_avg = roc_raw.get("micro_avg") or {}
            micro_auc: float | None = None
            micro_fpr = roc_micro_avg.get("fpr")
            micro_tpr = roc_micro_avg.get("tpr")
            micro_thresholds = roc_micro_avg.get("thresholds")

            if micro_fpr is not None and micro_tpr is not None and micro_thresholds is not None:
                micro_auc = safe_float_scalar(roc_micro_avg.get("auc", float("nan")))
                curves.append(
                    {
                        "label": "micro",
                        "fpr": safe_float_list(micro_fpr),
                        "tpr": safe_float_list(micro_tpr),
                        "thresholds": safe_float_list(micro_thresholds),
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

    if decoder_payload is not None:
        result["decoder_outputs"] = decoder_payload

    ensemble_report = None
    if voting_cls_acc is not None:
        ensemble_report = voting_cls_acc.finalize()
        result["ensemble_report"] = ensemble_report
    elif voting_reg_acc is not None:
        ensemble_report = voting_reg_acc.finalize()
        result["ensemble_report"] = ensemble_report
    elif bagging_cls_acc is not None:
        ensemble_report = bagging_cls_acc.finalize()
        result["ensemble_report"] = ensemble_report
    elif bagging_reg_acc is not None:
        ensemble_report = bagging_reg_acc.finalize()
        result["ensemble_report"] = ensemble_report
    elif adaboost_cls_acc is not None:
        ensemble_report = adaboost_cls_acc.finalize()
        result["ensemble_report"] = ensemble_report
    elif adaboost_reg_acc is not None:
        ensemble_report = adaboost_reg_acc.finalize()
        result["ensemble_report"] = ensemble_report
    elif xgb_acc is not None:
        ensemble_report = xgb_acc.finalize()
        result["ensemble_report"] = ensemble_report

    # --- Feature-related notes ----------------------------------------------
    if cfg.features.method == "pca":
        result["notes"].append("Model trained on PCA-transformed features.")
    if cfg.features.method == "lda":
        result["notes"].append("LDA was fitted on the training labels.")
    if cfg.features.method == "sfs":
        result["notes"].append("SFS performed wrapper-based feature selection on training data.")

    # --- Artifact metadata ---------------------------------------------------
    n_features_in = int(X.shape[1]) if hasattr(X, "shape") and len(X.shape) > 1 else None
    classes_out = result["confusion"]["labels"] if result.get("confusion") and result["confusion"]["labels"] else None

    # Extra stats for artifact meta (supports multiple ensemble kinds)
    extra_stats = None
    if ensemble_report and isinstance(ensemble_report, dict):
        kind = ensemble_report.get("kind")
        if kind == "voting":
            task = (ensemble_report.get("task") or "classification")
            if task == "regression":
                sim = ensemble_report.get("similarity") or {}
                errs = ensemble_report.get("errors") or {}
                ens_err = (errs.get("ensemble") or {})
                gain = (errs.get("gain_vs_best") or {})
                extra_stats = {
                    "ensemble_kind": "voting",
                    "ensemble_task": "regression",
                    "ensemble_n_estimators": ensemble_report.get("n_estimators"),
                    "ensemble_pairwise_mean_corr": sim.get("pairwise_mean_corr"),
                    "ensemble_pairwise_mean_absdiff": sim.get("pairwise_mean_absdiff"),
                    "ensemble_prediction_spread": sim.get("prediction_spread_mean"),
                    "ensemble_rmse": ens_err.get("rmse"),
                    "ensemble_mae": ens_err.get("mae"),
                    "ensemble_best_estimator": (ensemble_report.get("best_estimator") or {}).get("name"),
                    "ensemble_rmse_reduction_vs_best": gain.get("rmse_reduction"),
                    "ensemble_mae_reduction_vs_best": gain.get("mae_reduction"),
                }
            else:
                extra_stats = {
                    "ensemble_kind": "voting",
                    "ensemble_task": "classification",
                    "ensemble_n_estimators": ensemble_report.get("n_estimators"),
                    "ensemble_all_agree_rate": (ensemble_report.get("agreement") or {}).get("all_agree_rate"),
                    "ensemble_pairwise_agreement": (ensemble_report.get("agreement") or {}).get("pairwise_mean_agreement"),
                    "ensemble_tie_rate": (ensemble_report.get("vote") or {}).get("tie_rate"),
                    "ensemble_mean_margin": (ensemble_report.get("vote") or {}).get("mean_margin"),
                    "ensemble_best_estimator": (ensemble_report.get("best_estimator") or {}).get("name"),
                    "ensemble_corrected_vs_best": (ensemble_report.get("change_vs_best") or {}).get("corrected"),
                    "ensemble_harmed_vs_best": (ensemble_report.get("change_vs_best") or {}).get("harmed"),
                }
        elif kind == "bagging":
            task = (ensemble_report.get("task") or "classification")
            if task == "regression":
                sim = ensemble_report.get("diversity") or {}
                errs = ensemble_report.get("errors") or {}
                ens_err = (errs.get("ensemble") or {})
                gain = (errs.get("gain_vs_best") or {})
                extra_stats = {
                    "ensemble_kind": "bagging",
                    "ensemble_task": "regression",
                    "ensemble_n_estimators": (ensemble_report.get("bagging") or {}).get("n_estimators"),
                    "ensemble_base_algo": (ensemble_report.get("bagging") or {}).get("base_algo"),
                    "ensemble_oob_score": (ensemble_report.get("oob") or {}).get("score"),
                    "ensemble_oob_coverage": (ensemble_report.get("oob") or {}).get("coverage_rate"),
                    "ensemble_pairwise_mean_corr": sim.get("pairwise_mean_corr"),
                    "ensemble_pairwise_mean_absdiff": sim.get("pairwise_mean_absdiff"),
                    "ensemble_prediction_spread": sim.get("prediction_spread_mean"),
                    "ensemble_rmse": ens_err.get("rmse"),
                    "ensemble_mae": ens_err.get("mae"),
                    "ensemble_r2": ens_err.get("r2"),
                    "ensemble_rmse_reduction_vs_best": gain.get("rmse"),
                    "ensemble_mae_reduction_vs_best": gain.get("mae"),
                    "ensemble_r2_gain_vs_best": gain.get("r2"),
                }
            else:
                extra_stats = {
                    "ensemble_kind": "bagging",
                    "ensemble_task": "classification",
                    "ensemble_n_estimators": (ensemble_report.get("bagging") or {}).get("n_estimators"),
                    "ensemble_base_algo": (ensemble_report.get("bagging") or {}).get("base_algo"),
                    "ensemble_oob_score": (ensemble_report.get("oob") or {}).get("score"),
                    "ensemble_oob_coverage": (ensemble_report.get("oob") or {}).get("coverage_rate"),
                    "ensemble_all_agree_rate": (ensemble_report.get("diversity") or {}).get("all_agree_rate"),
                    "ensemble_pairwise_agreement": (ensemble_report.get("diversity") or {}).get("pairwise_mean_agreement"),
                    "ensemble_tie_rate": (ensemble_report.get("vote") or {}).get("tie_rate"),
                    "ensemble_mean_margin": (ensemble_report.get("vote") or {}).get("mean_margin"),
                }
        elif kind == "adaboost":
            task = (ensemble_report.get("task") or "classification")
            if task == "regression":
                errs = ensemble_report.get("errors") or {}
                ens_err = (errs.get("ensemble") or {})
                gain = (errs.get("gain_vs_best") or {})
                extra_stats = {
                    "ensemble_kind": "adaboost",
                    "ensemble_task": "regression",
                    "ensemble_n_estimators": (ensemble_report.get("adaboost") or {}).get("n_estimators"),
                    "ensemble_learning_rate": (ensemble_report.get("adaboost") or {}).get("learning_rate"),
                    "ensemble_effective_n": (ensemble_report.get("weights") or {}).get("effective_n_mean"),
                    "ensemble_rmse": ens_err.get("rmse"),
                    "ensemble_mae": ens_err.get("mae"),
                    "ensemble_rmse_reduction_vs_best": gain.get("rmse"),
                    "ensemble_mae_reduction_vs_best": gain.get("mae"),
                }
            else:
                extra_stats = {
                    "ensemble_kind": "adaboost",
                    "ensemble_task": "classification",
                    "ensemble_n_estimators": (ensemble_report.get("adaboost") or {}).get("n_estimators"),
                    "ensemble_learning_rate": (ensemble_report.get("adaboost") or {}).get("learning_rate"),
                    "ensemble_tie_rate": (ensemble_report.get("vote") or {}).get("tie_rate"),
                    "ensemble_mean_margin": (ensemble_report.get("vote") or {}).get("mean_margin"),
                    "ensemble_effective_n": (ensemble_report.get("weights") or {}).get("effective_n_mean"),
                }
        elif kind == "xgboost":
            extra_stats = {
                "ensemble_kind": "xgboost",
                "ensemble_best_iteration": (ensemble_report.get("xgboost") or {}).get("best_iteration_mean"),
                "ensemble_best_score": (ensemble_report.get("xgboost") or {}).get("best_score_mean"),
                "ensemble_n_features_seen": (ensemble_report.get("feature_importance") or {}).get("n_features_seen"),
            }

    artifact_input = ArtifactBuilderInput(
        cfg=cfg,
        pipeline=effective_model,
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
            "extra_stats": extra_stats,
        },
    )

    result["artifact"] = build_model_artifact_meta(artifact_input)
    try:
        uid = result["artifact"]["uid"]
        artifact_cache.put(uid, effective_model)
        # Cache eval outputs for post-hoc export (decoder CSV).
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

    # --- Shuffle baseline ----------------------------------------------------
    n_shuffles = int(getattr(cfg.eval, "n_shuffles", 0) or 0)
    if n_shuffles > 0:
        result["notes"].append("Shuffle baseline is not yet supported for ensembles.")

    from engine.contracts.results.ensemble import EnsembleResult

    return EnsembleResult.model_validate(result).model_dump()
