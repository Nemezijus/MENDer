import numpy as np
from typing import Dict, Any, Optional, Sequence

from shared_schemas.ensemble_run_config import EnsembleRunConfig

from utils.factories.data_loading_factory import make_data_loader
from utils.factories.sanity_factory import make_sanity_checker
from utils.factories.split_factory import make_splitter
from utils.factories.eval_factory import make_evaluator
from utils.factories.metrics_factory import make_metrics_computer
from utils.factories.ensemble_factory import make_ensemble_strategy
from utils.permutations.rng import RngManager
from utils.persistence.modelArtifact import ArtifactBuilderInput, build_model_artifact_meta
from utils.persistence.artifact_cache import artifact_cache
from utils.postprocessing.scoring import PROBA_METRICS

from ..adapters.io_adapter import LoadError

from sklearn.ensemble import VotingClassifier, VotingRegressor
from shared_schemas.ensemble_configs import (
    VotingEnsembleConfig,
    BaggingEnsembleConfig,
    AdaBoostEnsembleConfig,
    XGBoostEnsembleConfig,
)

from utils.postprocessing.ensembles.voting_ensemble_reporting import (
    VotingEnsembleReportAccumulator,
    VotingEnsembleRegressorReportAccumulator,
)
from utils.postprocessing.ensembles.bagging_ensemble_reporting import (
    BaggingEnsembleReportAccumulator,
    BaggingEnsembleRegressorReportAccumulator,
)
from utils.postprocessing.ensembles.adaboost_ensemble_reporting import (
    AdaBoostEnsembleReportAccumulator,
    AdaBoostEnsembleRegressorReportAccumulator,
)
from utils.postprocessing.ensembles.xgboost_ensemble_reporting import XGBoostEnsembleReportAccumulator

from utils.postprocessing.decoder_outputs import compute_decoder_outputs
from utils.predicting.prediction_results import build_decoder_output_table



def _safe_float_list(arr) -> list[float]:
    """Convert array-like to a JSON-safe list of finite floats."""
    a = np.asarray(arr, dtype=float)
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    return a.tolist()


def _safe_float_scalar(x: float) -> float:
    """Make a single float JSON-safe (no NaN/inf)."""
    return float(np.nan_to_num(float(x), nan=0.0, posinf=1.0, neginf=0.0))



def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """De-duplicate while preserving order (useful for repeated decoder notes across folds)."""
    out: list[str] = []
    seen: set[str] = set()
    for s in items:
        ss = str(s)
        if ss in seen:
            continue
        seen.add(ss)
        out.append(ss)
    return out


def _unwrap_final_estimator(model):
    """
    If model is a sklearn Pipeline, return its last step estimator.
    Otherwise return model unchanged.
    """
    try:
        if hasattr(model, "steps") and isinstance(getattr(model, "steps"), list) and len(model.steps) > 0:
            return model.steps[-1][1]
    except Exception:
        pass
    return model


def _transform_through_pipeline(model, X):
    """
    If model is a fitted sklearn Pipeline, transform X through all steps except the last estimator.
    If any step cannot transform, return X unchanged.
    """
    try:
        if not (hasattr(model, "steps") and isinstance(getattr(model, "steps"), list) and len(model.steps) > 1):
            return X

        Xt = X
        for _, step in model.steps[:-1]:
            if step is None or step == "passthrough":
                continue
            if hasattr(step, "transform"):
                Xt = step.transform(Xt)
            else:
                # Can't safely transform (no refit). Fall back to raw X.
                return X
        return Xt
    except Exception:
        return X


def _slice_X_by_features(X, feat_idx):
    # Best-effort column slicing that works for numpy arrays, pandas DataFrames, and scipy sparse.
    if feat_idx is None:
        return X
    try:
        idx = feat_idx
        # sklearn uses numpy arrays of indices; ensure list-like works
        if not isinstance(idx, (slice, list, tuple)):
            idx = list(idx)
    except Exception:
        idx = feat_idx
    try:
        return X[:, idx]
    except Exception:
        try:
            import numpy as _np
            Xa = _np.asarray(X)
            return Xa[:, idx]
        except Exception:
            return X


def _get_classes_arr(model) -> Optional[np.ndarray]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    try:
        arr = np.asarray(classes)
        return arr if arr.size > 0 else None
    except Exception:
        return None


def _should_decode_from_index_space(y_true: np.ndarray, y_pred_raw: np.ndarray, classes_arr: Optional[np.ndarray]) -> bool:
    """
    Deterministic decode decision:
      - we only decode if predictions look like integer indices in [0, K-1]
      - AND those indices are not already valid class labels (e.g. classes are 0..K-1)

    This avoids decoding when user classes are already 0..K-1.
    """
    if classes_arr is None:
        return False

    K = int(len(classes_arr))
    if K <= 0:
        return False

    a = np.asarray(y_pred_raw)
    if a.size == 0:
        return False

    # must be integer-ish indices
    if not np.issubdtype(a.dtype, np.integer):
        return False

    mn = int(a.min())
    mx = int(a.max())
    if mn < 0 or mx >= K:
        return False

    # If predicted values are already among classes_ values (e.g. classes_ are [0,1,2]),
    # then decoding would be wrong.
    try:
        pred_set = set(np.unique(a).tolist())
        class_set = set(np.unique(classes_arr).tolist())
        if pred_set.issubset(class_set):
            return False
    except Exception:
        pass

    # Also check: y_true values match the declared classes (common case)
    # If they do, decoding indices via classes_ is meaningful.
    try:
        yt_set = set(np.unique(y_true).tolist())
        class_set = set(np.unique(classes_arr).tolist())
        if not yt_set.issubset(class_set):
            # y_true doesn't match model.classes_ (unexpected) -> don't decode
            return False
    except Exception:
        return False

    return True


def _encode_y_true_to_index(y_true: np.ndarray, classes_arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Deterministically map y_true values to 0..K-1 using model.classes_ ordering.
    Returns None if mapping not possible.
    """
    if classes_arr is None:
        return None
    try:
        idx_map = {cls: i for i, cls in enumerate(classes_arr.tolist())}
        return np.asarray([idx_map[v] for v in y_true], dtype=int)
    except Exception:
        return None


def _friendly_ensemble_training_error(
    e: Exception, cfg: EnsembleRunConfig, *, fold_id: int | None = None
) -> ValueError:
    """
    Convert common low-level sklearn/3rd-party errors into actionable, user-friendly messages.

    We return ValueError so the router can surface a 400 with a clear explanation.
    """
    kind = getattr(cfg.ensemble, "kind", "ensemble")
    msg = (str(e) or e.__class__.__name__).strip()
    fold_txt = f" (fold {fold_id})" if fold_id is not None else ""

    # --- Bagging / bootstrap: single-class resample -------------------------------
    if "needs samples of at least 2 classes" in msg and "only one class" in msg:
        return ValueError(
            "Bagging training failed because at least one bootstrap sample contained only a single class"
            f"{fold_txt}.\n"
            "Some base estimators (e.g., Logistic Regression, SVM) cannot be fit on single-class data.\n\n"
            "Try one of the following:\n"
            "• Enable Balanced bagging (recommended)\n"
            "• Increase max_samples (use larger bags)\n"
            "• Disable bootstrap (bootstrap=false)\n"
            "• Use a tree-based base estimator\n\n"
            f"Raw error: {msg}"
        )

    # --- Balanced bagging dependency missing --------------------------------------
    if isinstance(e, ImportError) or "imbalanced-learn" in msg.lower():
        return ValueError(
            "Balanced bagging requires the optional dependency `imbalanced-learn`.\n"
            "Install it in your environment (and rebuild Docker if applicable), then retry.\n\n"
            f"Raw error: {msg}"
        )

    # --- AdaBoost + pipeline sample_weight routing --------------------------------
    if "Pipeline doesn't support sample_weight" in msg:
        return ValueError(
            "AdaBoost training failed because the current preprocessing pipeline cannot accept sample weights.\n"
            "AdaBoost reweights samples at each boosting stage and requires the estimator fit() to support "
            "sample_weight.\n"
            "In MENDer, we address this by fitting preprocessing once, transforming X, then boosting on "
            "transformed features.\n\n"
            f"Raw error: {msg}"
        )

    # --- XGBoost label encoding ----------------------------------------------------
    if "Invalid classes inferred from unique values of `y`" in msg:
        return ValueError(
            "XGBoost classification requires class labels encoded as 0..K-1.\n"
            "MENDer encodes labels internally for training and decodes predictions back to original labels.\n\n"
            f"Raw error: {msg}"
        )

    # Fallback: keep original message but add context.
    return ValueError(f"{kind} training failed{fold_txt}: {msg}")


def _extract_base_estimator_algo_from_cfg(cfg: EnsembleRunConfig, default: str = "default") -> str:
    """Best-effort: get algo name for Bagging base estimator from config."""
    try:
        be = getattr(cfg.ensemble, "base_estimator", None)
        if be is None:
            return default
        return str(getattr(be, "algo", default) or default)
    except Exception:
        return default


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
    is_xgboost_report = eval_kind == "classification" and isinstance(cfg.ensemble, XGBoostEnsembleConfig)

    for fold_id, (Xtr, Xte, ytr, yte) in enumerate(splitter.split(X, y), start=1):
        try:
            model = ensemble_strategy.fit(
                Xtr,
                ytr,
                rngm=rngm,
                stream=f"{mode}/fold{fold_id}",
            )
            last_model = model

            y_pred = model.predict(Xte)

            # --- Optional decoder outputs (classification only) ----------------
            if decoder_enabled and eval_kind == "classification":
                # Always track fold ID per test row for UI/diagnostics.
                n_fold_rows = int(np.asarray(y_pred).shape[0])
                decoder_fold_ids.append(np.full((n_fold_rows,), fold_id, dtype=int))

                try:
                    dec = compute_decoder_outputs(
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
            try:
                final_est = _unwrap_final_estimator(model)

                # --- classification voting report ---
                if eval_kind == "classification" and isinstance(final_est, VotingClassifier):
                    if voting_cls_acc is None:
                        est_names = [n for n, _ in getattr(final_est, "estimators", [])]
                        est_algos = [getattr(s.model, "algo", "model") for s in cfg.ensemble.estimators]
                        if len(est_algos) != len(est_names):
                            est_algos = (est_algos + ["model"] * len(est_names))[: len(est_names)]

                        voting_cls_acc = VotingEnsembleReportAccumulator.create(
                            estimator_names=est_names,
                            estimator_algos=est_algos,
                            metric_name=str(cfg.eval.metric),
                            weights=getattr(final_est, "weights", None),
                            voting=str(getattr(cfg.ensemble, "voting", "hard")),
                        )

                    base_preds: Dict[str, Any] = {}
                    base_scores: Dict[str, float] = {}

                    # VotingClassifier internally label-encodes y during fit().
                    yte_arr = np.asarray(yte)
                    yte_enc = yte_arr
                    le = getattr(final_est, "le_", None)
                    if le is not None:
                        try:
                            yte_enc = le.transform(yte_arr)
                        except Exception:
                            yte_enc = yte_arr

                    est_pairs = list(zip(getattr(final_est, "estimators", []), getattr(final_est, "estimators_", [])))
                    for (name, _unfitted), fitted in est_pairs:
                        yp_enc = fitted.predict(Xte)

                        yp_report = yp_enc
                        if le is not None:
                            try:
                                yp_report = le.inverse_transform(np.asarray(yp_enc))
                            except Exception:
                                yp_report = yp_enc

                        base_preds[name] = yp_report

                        y_proba_i = None
                        y_score_i = None
                        if hasattr(fitted, "predict_proba"):
                            try:
                                y_proba_i = fitted.predict_proba(Xte)
                            except Exception:
                                y_proba_i = None
                        if y_proba_i is None and hasattr(fitted, "decision_function"):
                            try:
                                y_score_i = fitted.decision_function(Xte)
                            except Exception:
                                y_score_i = None

                        base_scores[name] = float(
                            evaluator.score(
                                yte_enc,
                                y_pred=yp_enc,
                                y_proba=y_proba_i,
                                y_score=y_score_i,
                            )
                        )

                    if voting_cls_acc is not None:
                        voting_cls_acc.add_fold(
                            y_true=np.asarray(yte),
                            y_ensemble_pred=np.asarray(y_pred),
                            base_preds={k: np.asarray(v) for k, v in base_preds.items()},
                            base_scores=base_scores,
                        )

                # --- regression voting report ---
                elif eval_kind == "regression" and isinstance(final_est, VotingRegressor):
                    if voting_reg_acc is None:
                        est_names = [n for n, _ in getattr(final_est, "estimators", [])]
                        est_algos = [getattr(s.model, "algo", "model") for s in cfg.ensemble.estimators]
                        if len(est_algos) != len(est_names):
                            est_algos = (est_algos + ["model"] * len(est_names))[: len(est_names)]

                        voting_reg_acc = VotingEnsembleRegressorReportAccumulator.create(
                            estimator_names=est_names,
                            estimator_algos=est_algos,
                            metric_name=str(cfg.eval.metric),
                            weights=getattr(final_est, "weights", None),
                        )

                    base_preds: Dict[str, Any] = {}
                    base_scores: Dict[str, float] = {}

                    est_pairs = list(zip(getattr(final_est, "estimators", []), getattr(final_est, "estimators_", [])))
                    for (name, _unfitted), fitted in est_pairs:
                        yp = fitted.predict(Xte)
                        base_preds[name] = np.asarray(yp)
                        base_scores[name] = float(
                            evaluator.score(
                                np.asarray(yte),
                                y_pred=np.asarray(yp),
                                y_proba=None,
                                y_score=None,
                            )
                        )

                    if voting_reg_acc is not None:
                        voting_reg_acc.add_fold(
                            y_true=np.asarray(yte),
                            y_ensemble_pred=np.asarray(y_pred),
                            base_preds={k: np.asarray(v) for k, v in base_preds.items()},
                            base_scores=base_scores,
                        )
            except Exception:
                pass


# ---------------- Bagging report ----------------
        if is_bagging_report:
            try:
                ests = getattr(model, "estimators_", None)
                feats_list = getattr(model, "estimators_features_", None)

                if ests is not None and len(ests) > 0:
                    metric_name = str(cfg.eval.metric)
                    base_algo = _extract_base_estimator_algo_from_cfg(cfg, default="default")

                    # --- classification bagging report ---
                    if eval_kind == "classification":
                        if bagging_cls_acc is None:
                            bagging_cls_acc = BaggingEnsembleReportAccumulator.create(
                                metric_name=str(cfg.eval.metric),
                                base_algo=base_algo,
                                n_estimators=int(getattr(cfg.ensemble, "n_estimators", len(ests)) or len(ests)),
                                max_samples=getattr(cfg.ensemble, "max_samples", None),
                                max_features=getattr(cfg.ensemble, "max_features", None),
                                bootstrap=bool(getattr(cfg.ensemble, "bootstrap", True)),
                                bootstrap_features=bool(getattr(cfg.ensemble, "bootstrap_features", False)),
                                oob_score_enabled=bool(getattr(cfg.ensemble, "oob_score", False)),
                                balanced=bool(getattr(cfg.ensemble, "balanced", False)),
                                sampling_strategy=getattr(cfg.ensemble, "sampling_strategy", None),
                                replacement=getattr(cfg.ensemble, "replacement", None),
                            )

                        base_pred_cols = []
                        base_scores_list: list[float] = []

                        classes_arr = _get_classes_arr(model)
                        yte_arr = np.asarray(yte)
                        yte_enc = _encode_y_true_to_index(yte_arr, classes_arr)

                        for i, est in enumerate(ests):
                            if est is None:
                                continue

                            feat_idx = None
                            try:
                                if feats_list is not None and i < len(feats_list):
                                    feat_idx = feats_list[i]
                            except Exception:
                                feat_idx = None

                            Xte_i = _slice_X_by_features(Xte, feat_idx)

                            # IMPORTANT: if max_features < 1.0, each base estimator was trained on a subset.
                            # We must apply the SAME subset at predict/score time.
                            yp_raw = np.asarray(est.predict(Xte_i))

                            if _should_decode_from_index_space(yte_arr, yp_raw, classes_arr):
                                yp_dec = classes_arr[yp_raw.astype(int, copy=False)]
                            else:
                                yp_dec = yp_raw

                            base_pred_cols.append(np.asarray(yp_dec))

                            # score distribution (handle PROBA metrics deterministically in encoded space)
                            try:
                                y_proba_i = None
                                y_score_i = None

                                if metric_name in PROBA_METRICS:
                                    if hasattr(est, "predict_proba"):
                                        try:
                                            y_proba_i = est.predict_proba(Xte_i)
                                        except Exception:
                                            y_proba_i = None
                                    if y_proba_i is None and hasattr(est, "decision_function"):
                                        try:
                                            y_score_i = est.decision_function(Xte_i)
                                        except Exception:
                                            y_score_i = None

                                    if y_proba_i is None and y_score_i is None:
                                        s = None
                                    elif yte_enc is not None and _should_decode_from_index_space(yte_arr, yp_raw, classes_arr):
                                        s = evaluator.score(
                                            yte_enc,
                                            y_pred=yp_raw,
                                            y_proba=y_proba_i,
                                            y_score=y_score_i,
                                        )
                                    else:
                                        s = evaluator.score(
                                            yte_arr,
                                            y_pred=yp_dec,
                                            y_proba=y_proba_i,
                                            y_score=y_score_i,
                                        )
                                else:
                                    s = evaluator.score(
                                        yte_arr,
                                        y_pred=yp_dec,
                                        y_proba=None,
                                        y_score=None,
                                    )

                                if s is not None:
                                    base_scores_list.append(float(s))
                            except Exception:
                                pass

                        if base_pred_cols and bagging_cls_acc is not None:
                            base_preds_mat = np.column_stack(base_pred_cols)

                            oob_score = getattr(model, "oob_score_", None)
                            oob_decision = getattr(model, "oob_decision_function_", None)

                            bagging_cls_acc.add_fold(
                                base_preds=base_preds_mat,
                                oob_score=oob_score if oob_score is not None else None,
                                oob_decision_function=oob_decision,
                                base_estimator_scores=base_scores_list if base_scores_list else None,
                            )

                    # --- regression bagging report ---
                    elif eval_kind == "regression":
                        if bagging_reg_acc is None:
                            bagging_reg_acc = BaggingEnsembleRegressorReportAccumulator.create(
                                metric_name=str(cfg.eval.metric),
                                base_algo=base_algo,
                                n_estimators=int(getattr(cfg.ensemble, "n_estimators", len(ests)) or len(ests)),
                                max_samples=getattr(cfg.ensemble, "max_samples", None),
                                max_features=getattr(cfg.ensemble, "max_features", None),
                                bootstrap=bool(getattr(cfg.ensemble, "bootstrap", True)),
                                bootstrap_features=bool(getattr(cfg.ensemble, "bootstrap_features", False)),
                                oob_score_enabled=bool(getattr(cfg.ensemble, "oob_score", False)),
                            )

                        base_pred_cols = []
                        base_scores_list: list[float] = []

                        yte_arr = np.asarray(yte, dtype=float)
                        yens = np.asarray(y_pred, dtype=float)

                        for i, est in enumerate(ests):
                            if est is None:
                                continue

                            feat_idx = None
                            try:
                                if feats_list is not None and i < len(feats_list):
                                    feat_idx = feats_list[i]
                            except Exception:
                                feat_idx = None

                            Xte_i = _slice_X_by_features(Xte, feat_idx)

                            yp = np.asarray(est.predict(Xte_i), dtype=float)
                            base_pred_cols.append(yp)

                            try:
                                s = evaluator.score(
                                    yte_arr,
                                    y_pred=yp,
                                    y_proba=None,
                                    y_score=None,
                                )
                                base_scores_list.append(float(s))
                            except Exception:
                                pass

                        if base_pred_cols and bagging_reg_acc is not None:
                            base_preds_mat = np.column_stack(base_pred_cols)

                            oob_score = getattr(model, "oob_score_", None)
                            oob_pred = getattr(model, "oob_prediction_", None)

                            bagging_reg_acc.add_fold(
                                y_true=yte_arr,
                                ensemble_pred=yens,
                                base_preds=base_preds_mat,
                                oob_score=oob_score if oob_score is not None else None,
                                oob_prediction=oob_pred,
                                base_estimator_scores=base_scores_list if base_scores_list else None,
                            )
            except Exception:
                pass

        # ---------------- AdaBoost report (classification + regression) ----------------
        if is_adaboost_report:
            try:
                # AdaBoost often comes wrapped as a Pipeline(pre -> clf). In this repo, step names
                # may be ('pre','clf') or ('scale','feat','clf'). We handle both.
                boost = None
                Xte_boost = Xte

                # Prefer named_steps if present (most robust for your setup)
                if hasattr(model, "named_steps") and isinstance(getattr(model, "named_steps"), dict):
                    # try "clf" for the boosting estimator
                    clf_step = model.named_steps.get("clf", None)
                    if clf_step is not None:
                        boost = clf_step

                    # try "pre" as the preprocessor (if you used that naming)
                    pre_step = model.named_steps.get("pre", None)
                    if pre_step is not None and hasattr(pre_step, "transform"):
                        try:
                            Xte_boost = pre_step.transform(Xte)
                        except Exception:
                            Xte_boost = Xte
                    else:
                        # otherwise transform through all non-final steps
                        Xte_boost = _transform_through_pipeline(model, Xte)

                if boost is None:
                    # Fall back: last estimator of pipeline OR model itself
                    boost = _unwrap_final_estimator(model)
                    if boost is not model:
                        Xte_boost = _transform_through_pipeline(model, Xte)

                # If you ever wrap AdaBoost (similar to XGB label adapter), unwrap it
                boost = getattr(boost, "model", boost)

                ests = getattr(boost, "estimators_", None)
                w = getattr(boost, "estimator_weights_", None)
                errs = getattr(boost, "estimator_errors_", None)

                m = len(ests) if ests is not None else 0
                if w is not None and m > 0:
                    w = np.asarray(w, dtype=float)[:m]
                if errs is not None and m > 0:
                    errs = np.asarray(errs, dtype=float)[:m]

                if ests is not None and len(ests) > 0 and w is not None:
                    base_algo = _extract_base_estimator_algo_from_cfg(cfg, default="default")
                    metric_name = str(cfg.eval.metric)

                    # --- classification adaboost report ---
                    if eval_kind == "classification":
                        if adaboost_cls_acc is None:
                            adaboost_cls_acc = AdaBoostEnsembleReportAccumulator.create(
                                metric_name=str(cfg.eval.metric),
                                base_algo=base_algo,
                                n_estimators=int(getattr(cfg.ensemble, "n_estimators", len(ests)) or len(ests)),
                                learning_rate=float(getattr(cfg.ensemble, "learning_rate", 1.0) or 1.0),
                                algorithm=str(getattr(cfg.ensemble, "algorithm", None) or None),
                            )

                        classes_arr = _get_classes_arr(boost)
                        if classes_arr is None:
                            classes_arr = _get_classes_arr(model)
                        yte_arr = np.asarray(yte)
                        yte_enc = _encode_y_true_to_index(yte_arr, classes_arr) if classes_arr is not None else None

                        base_pred_cols = []
                        base_scores_list: list[float] = []

                        for est in ests:
                            if est is None:
                                continue

                            yp_raw = np.asarray(est.predict(Xte_boost))

                            # Deterministic decode protection (same as bagging)
                            if _should_decode_from_index_space(yte_arr, yp_raw, classes_arr):
                                yp_dec = classes_arr[yp_raw.astype(int, copy=False)]
                            else:
                                yp_dec = yp_raw

                            base_pred_cols.append(np.asarray(yp_dec))

                            # Optional per-stage score distribution (best-effort)
                            try:
                                y_proba_i = None
                                y_score_i = None

                                if metric_name in PROBA_METRICS:
                                    if hasattr(est, "predict_proba"):
                                        try:
                                            y_proba_i = est.predict_proba(Xte_boost)
                                        except Exception:
                                            y_proba_i = None
                                    if y_proba_i is None and hasattr(est, "decision_function"):
                                        try:
                                            y_score_i = est.decision_function(Xte_boost)
                                        except Exception:
                                            y_score_i = None

                                    if y_proba_i is None and y_score_i is None:
                                        s = None
                                    elif yte_enc is not None and _should_decode_from_index_space(yte_arr, yp_raw, classes_arr):
                                        s = evaluator.score(
                                            yte_enc,
                                            y_pred=yp_raw,
                                            y_proba=y_proba_i,
                                            y_score=y_score_i,
                                        )
                                    else:
                                        s = evaluator.score(
                                            yte_arr,
                                            y_pred=yp_dec,
                                            y_proba=y_proba_i,
                                            y_score=y_score_i,
                                        )
                                else:
                                    s = evaluator.score(
                                        yte_arr,
                                        y_pred=yp_dec,
                                        y_proba=None,
                                        y_score=None,
                                    )

                                if s is not None:
                                    base_scores_list.append(float(s))
                            except Exception:
                                pass

                        if base_pred_cols and adaboost_cls_acc is not None:
                            base_preds_mat = np.column_stack(base_pred_cols)

                            # ---- stage/weight diagnostics (configured vs fitted vs effective) ----
                            w_arr = np.asarray(w, dtype=float)
                            weight_eps = 1e-6

                            n_estimators_fitted = int(len(ests)) if ests is not None else int(base_preds_mat.shape[1])
                            n_nonzero_weights = int(np.sum(w_arr > 0))
                            n_nontrivial_weights = int(np.sum(w_arr > weight_eps))

                            weight_mass_topk = None
                            try:
                                ssum = float(np.sum(w_arr))
                                if ssum > 0:
                                    w_sorted = np.sort(w_arr)[::-1]
                                    c = np.cumsum(w_sorted) / ssum
                                    topks = [5, 10, 20]
                                    weight_mass_topk = {k: float(c[min(k, len(c)) - 1]) for k in topks if len(c) > 0}
                            except Exception:
                                weight_mass_topk = None

                            adaboost_cls_acc.add_fold(
                                base_preds=base_preds_mat,
                                estimator_weights=np.asarray(w, dtype=float)[: base_preds_mat.shape[1]],
                                estimator_errors=np.asarray(errs, dtype=float)[: base_preds_mat.shape[1]] if errs is not None else None,
                                base_estimator_scores=base_scores_list if base_scores_list else None,

                                # diagnostics
                                n_estimators_fitted=n_estimators_fitted,
                                n_nonzero_weights=n_nonzero_weights,
                                n_nontrivial_weights=n_nontrivial_weights,
                                weight_eps=weight_eps,
                                weight_mass_topk=weight_mass_topk,
                            )

                    # --- regression adaboost report ---
                    elif eval_kind == "regression":
                        if adaboost_reg_acc is None:
                            adaboost_reg_acc = AdaBoostEnsembleRegressorReportAccumulator.create(
                                metric_name=str(cfg.eval.metric),
                                base_algo=base_algo,
                                n_estimators=int(getattr(cfg.ensemble, "n_estimators", len(ests)) or len(ests)),
                                learning_rate=float(getattr(cfg.ensemble, "learning_rate", 1.0) or 1.0),
                                loss=None,
                            )

                        yte_arr = np.asarray(yte, dtype=float)
                        yens = np.asarray(y_pred, dtype=float)

                        base_pred_cols = []
                        base_scores_list: list[float] = []

                        for est in ests:
                            if est is None:
                                continue

                            yp = np.asarray(est.predict(Xte_boost), dtype=float)
                            base_pred_cols.append(yp)

                            try:
                                s = evaluator.score(
                                    yte_arr,
                                    y_pred=yp,
                                    y_proba=None,
                                    y_score=None,
                                )
                                base_scores_list.append(float(s))
                            except Exception:
                                pass

                        if base_pred_cols and adaboost_reg_acc is not None:
                            base_preds_mat = np.column_stack(base_pred_cols)

                            w_arr = np.asarray(w, dtype=float)
                            weight_eps = 1e-6

                            n_estimators_fitted = int(len(ests)) if ests is not None else int(base_preds_mat.shape[1])
                            n_nonzero_weights = int(np.sum(w_arr > 0))
                            n_nontrivial_weights = int(np.sum(w_arr > weight_eps))

                            weight_mass_topk = None
                            try:
                                ssum = float(np.sum(w_arr))
                                if ssum > 0:
                                    w_sorted = np.sort(w_arr)[::-1]
                                    c = np.cumsum(w_sorted) / ssum
                                    topks = [5, 10, 20]
                                    weight_mass_topk = {k: float(c[min(k, len(c)) - 1]) for k in topks if len(c) > 0}
                            except Exception:
                                weight_mass_topk = None

                            adaboost_reg_acc.add_fold(
                                y_true=yte_arr,
                                ensemble_pred=yens,
                                base_preds=base_preds_mat,
                                estimator_weights=np.asarray(w, dtype=float)[: base_preds_mat.shape[1]],
                                estimator_errors=np.asarray(errs, dtype=float)[: base_preds_mat.shape[1]] if errs is not None else None,
                                base_estimator_scores=base_scores_list if base_scores_list else None,
                                n_estimators_fitted=n_estimators_fitted,
                                n_nonzero_weights=n_nonzero_weights,
                                n_nontrivial_weights=n_nontrivial_weights,
                                weight_eps=weight_eps,
                                weight_mass_topk=weight_mass_topk,
                            )

            except Exception:
                pass

        # ---------------- XGBoost report (adjusted) ----------------
        if is_xgboost_report:
            try:
                inner = _unwrap_final_estimator(model)

                # If pipeline-wrapped, prefer named_steps['clf'] (consistent with how you build pipelines)
                if hasattr(model, "named_steps") and isinstance(getattr(model, "named_steps"), dict):
                    clf_step = model.named_steps.get("clf", None)
                    if clf_step is not None:
                        inner = clf_step

                # Unwrap XGB label adapter: XGBClassifierLabelAdapter(model=..., ...)
                inner = getattr(inner, "model", inner)

                if xgb_acc is None:
                    params = {}
                    try:
                        params = dict(getattr(inner, "get_params", lambda: {})() or {})
                    except Exception:
                        params = {}

                    # XGBoost training eval metric used for eval_set curves (often "mlogloss"/"logloss"/"rmse")
                    train_eval_metric = None
                    try:
                        tem = params.get("eval_metric", None)
                        if isinstance(tem, (list, tuple)) and len(tem) > 0:
                            train_eval_metric = str(tem[0])
                        elif tem is not None:
                            train_eval_metric = str(tem)
                    except Exception:
                        train_eval_metric = None

                    xgb_acc = XGBoostEnsembleReportAccumulator.create(
                        metric_name=str(cfg.eval.metric),          # final metric (MENDer)
                        train_eval_metric=train_eval_metric,       # training metric (XGB)
                        params=params,
                    )

                best_iteration = (
                    getattr(inner, "best_iteration", None)
                    or getattr(inner, "best_iteration_", None)
                )
                best_score = (
                    getattr(inner, "best_score", None)
                    or getattr(inner, "best_score_", None)
                )

                evals_result = getattr(inner, "evals_result_", None)
                if callable(evals_result):
                    evals_result = None
                if not evals_result:
                    best_iteration = None
                    best_score = None
                feat_imps = getattr(inner, "feature_importances_", None)

                feature_names = None
                try:
                    feature_names = None
                except Exception:
                    feature_names = None
                
                xgb_acc.add_fold(
                    best_iteration=best_iteration,
                    best_score=best_score,
                    evals_result=evals_result,
                    feature_importances=feat_imps,
                    feature_names=feature_names,
                )

            except Exception:
                pass

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

    y_proba_all_arr = np.concatenate(y_proba_all, axis=0) if y_proba_all else None
    y_score_all_arr = np.concatenate(y_score_all, axis=0) if y_score_all else None

    # --- Aggregate decoder outputs across folds (out-of-sample) -------------
    decoder_payload = None
    if decoder_enabled and eval_kind == "classification":
        try:
            ds_arr = np.concatenate(decoder_scores_all, axis=0) if decoder_scores_all else None
            pr_arr = np.concatenate(decoder_proba_all, axis=0) if decoder_proba_all else None
            mg_arr = np.concatenate(decoder_margin_all, axis=0) if decoder_margin_all else None
            fold_ids_arr = np.concatenate(decoder_fold_ids, axis=0) if decoder_fold_ids else None

            preview_rows = build_decoder_output_table(
                indices=list(range(len(y_pred_all_arr))),
                y_pred=y_pred_all_arr,
                classes=decoder_classes,
                decision_scores=ds_arr,
                proba=pr_arr,
                margin=mg_arr,
                positive_class_label=decoder_positive_label,
                positive_class_index=decoder_positive_index,
                y_true=y_true_all_arr,
                max_rows=decoder_max_preview_rows if decoder_max_preview_rows > 0 else None,
            )

            # Add fold_id column when available (kfold); for holdout, fold_id will be 1.
            if fold_ids_arr is not None and len(preview_rows) > 0:
                for i, r in enumerate(preview_rows):
                    try:
                        r["fold_id"] = int(fold_ids_arr[i])
                    except Exception:
                        pass
            elif mode == "holdout" and len(preview_rows) > 0:
                for r in preview_rows:
                    r["fold_id"] = 1

            decoder_payload = {
                "classes": decoder_classes.tolist() if decoder_classes is not None else None,
                "positive_class_label": decoder_positive_label,
                "positive_class_index": decoder_positive_index,
                "has_decision_scores": ds_arr is not None,
                "has_proba": pr_arr is not None,
                "proba_source": decoder_proba_source,
                "notes": _dedupe_preserve_order(decoder_notes),
                "n_rows_total": int(len(y_pred_all_arr)),
                "preview_rows": preview_rows,
            }
        except Exception as e:
            decoder_payload = {
                "classes": decoder_classes.tolist() if decoder_classes is not None else None,
                "positive_class_label": decoder_positive_label,
                "positive_class_index": decoder_positive_index,
                "has_decision_scores": False,
                "has_proba": False,
                "notes": _dedupe_preserve_order(
                    decoder_notes + [f"decoder aggregation failed: {type(e).__name__}: {e}"]
                ),
                "n_rows_total": int(len(y_pred_all_arr)) if y_pred_all_arr is not None else None,
                "preview_rows": [],
            }

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
                        "fpr": _safe_float_list(entry["fpr"]),
                        "tpr": _safe_float_list(entry["tpr"]),
                        "thresholds": _safe_float_list(entry["thresholds"]),
                        "auc": _safe_float_scalar(entry["auc"]),
                    }
                )

            roc_macro_avg = roc_raw.get("macro_avg") or {}
            macro_auc = _safe_float_scalar(roc_macro_avg["auc"]) if "auc" in roc_macro_avg else None

            macro_fpr = roc_macro_avg.get("fpr")
            macro_tpr = roc_macro_avg.get("tpr")
            if macro_fpr is not None and macro_tpr is not None:
                curves.append(
                    {
                        "label": "macro",
                        "fpr": _safe_float_list(macro_fpr),
                        "tpr": _safe_float_list(macro_tpr),
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
    except Exception:
        pass

    # --- Shuffle baseline ----------------------------------------------------
    n_shuffles = int(getattr(cfg.eval, "n_shuffles", 0) or 0)
    if n_shuffles > 0:
        result["notes"].append("Shuffle baseline is not yet supported for ensembles.")

    return result
