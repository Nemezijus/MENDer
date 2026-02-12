from __future__ import annotations

import numpy as np

from engine.components.evaluation.scoring import PROBA_METRICS
from engine.components.prediction.decoder_extraction import compute_decoder_outputs_raw
from engine.reporting.ensembles.helpers import _friendly_ensemble_training_error
from engine.reporting.ensembles.reports.adaboost import update_adaboost_report
from engine.reporting.ensembles.reports.bagging import update_bagging_report
from engine.reporting.ensembles.reports.voting import update_voting_report
from engine.reporting.ensembles.reports.xgboost import update_xgboost_report

from .types import DecoderState, FoldState, ReportState


def run_ensemble_folds(
    *,
    cfg,
    X,
    y,
    splitter,
    ensemble_strategy,
    evaluator,
    eval_kind: str,
    mode: str,
    rngm,
    fold_state: FoldState,
    decoder_state: DecoderState,
    report_state: ReportState,
):
    """Run train/eval over all folds and update the provided states.

    Returns
    -------
    The model fitted on the last fold (for holdout this is the effective model).
    """

    last_model = None

    for fold_id, s in enumerate(splitter.split(X, y), start=1):
        Xtr, Xte, ytr, yte = s.Xtr, s.Xte, s.ytr, s.yte
        idx_te = s.idx_te

        if idx_te is not None:
            fold_state.test_indices_parts.append(np.asarray(idx_te, dtype=int).ravel())
        else:
            fold_state.test_indices_parts.append(None)

        try:
            model = ensemble_strategy.fit(
                Xtr,
                ytr,
                rngm=rngm,
                stream=f"{mode}/fold{fold_id}",
            )
            last_model = model

            y_pred = model.predict(Xte)

            # Fold id per evaluation row
            try:
                n_fold_rows = int(np.asarray(y_pred).shape[0])
            except Exception:
                n_fold_rows = int(np.asarray(Xte).shape[0])
            fold_state.eval_fold_ids_parts.append(np.full((n_fold_rows,), fold_id, dtype=int))

            # Optional decoder outputs (classification only)
            if decoder_state.enabled and eval_kind == "classification":
                decoder_state.fold_ids_parts.append(np.full((n_fold_rows,), fold_id, dtype=int))
                try:
                    dec = compute_decoder_outputs_raw(
                        model,
                        Xte,
                        positive_class_label=decoder_state.positive_label,
                        include_decision_scores=decoder_state.include_scores,
                        include_probabilities=decoder_state.include_probabilities,
                        calibrate_probabilities=decoder_state.calibrate_probabilities,
                    )
                    if decoder_state.classes is None and dec.classes is not None:
                        decoder_state.classes = np.asarray(dec.classes)
                    if decoder_state.positive_index is None and dec.positive_class_index is not None:
                        decoder_state.positive_index = int(dec.positive_class_index)

                    if dec.decision_scores is not None:
                        decoder_state.scores_all.append(np.asarray(dec.decision_scores))
                    if dec.proba is not None:
                        decoder_state.proba_all.append(np.asarray(dec.proba))
                    try:
                        src = getattr(dec, "proba_source", None)
                        if src is not None:
                            if decoder_state.proba_source is None:
                                decoder_state.proba_source = str(src)
                            elif str(src) != str(decoder_state.proba_source):
                                decoder_state.proba_source = "mixed"
                    except Exception:
                        pass

                    if decoder_state.include_margin and dec.margin is not None:
                        decoder_state.margin_all.append(np.asarray(dec.margin))
                    if dec.notes:
                        decoder_state.notes.extend([str(x) for x in dec.notes])
                except Exception as e:
                    decoder_state.notes.append(
                        f"decoder outputs failed on fold {fold_id}: {type(e).__name__}: {e}"
                    )

        except Exception as e:
            raise _friendly_ensemble_training_error(e, cfg, fold_id=fold_id) from e

        # --- Ensemble report updates (best-effort) --------------------------
        if report_state.is_voting:
            report_state.voting_cls_acc, report_state.voting_reg_acc = update_voting_report(
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
                voting_cls_acc=report_state.voting_cls_acc,
                voting_reg_acc=report_state.voting_reg_acc,
            )

        if report_state.is_bagging:
            report_state.bagging_cls_acc, report_state.bagging_reg_acc = update_bagging_report(
                cfg=cfg,
                eval_kind=eval_kind,
                model=model,
                Xte=Xte,
                yte=yte,
                y_pred=y_pred,
                fold_id=fold_id,
                evaluator=evaluator,
                bagging_cls_acc=report_state.bagging_cls_acc,
                bagging_reg_acc=report_state.bagging_reg_acc,
            )

        if report_state.is_adaboost:
            report_state.adaboost_cls_acc, report_state.adaboost_reg_acc = update_adaboost_report(
                cfg=cfg,
                eval_kind=eval_kind,
                model=model,
                Xte=Xte,
                yte=yte,
                y_pred=y_pred,
                fold_id=fold_id,
                evaluator=evaluator,
                adaboost_cls_acc=report_state.adaboost_cls_acc,
                adaboost_reg_acc=report_state.adaboost_reg_acc,
            )

        if report_state.is_xgboost:
            report_state.xgb_acc = update_xgboost_report(
                cfg=cfg,
                model=model,
                Xtr=Xtr,
                Xte=Xte,
                ytr=ytr,
                yte=yte,
                y_pred=y_pred,
                fold_id=fold_id,
                xgb_acc=report_state.xgb_acc,
            )

        # --- Score + pooled arrays -----------------------------------------
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

        score_val = evaluator.score(yte, y_pred=y_pred, y_proba=y_proba, y_score=y_score)
        fold_state.fold_scores.append(float(score_val))

        fold_state.y_true_parts.append(np.asarray(yte))
        fold_state.y_pred_parts.append(np.asarray(y_pred))
        if y_proba is not None:
            fold_state.y_proba_parts.append(np.asarray(y_proba))
        if y_score is not None:
            fold_state.y_score_parts.append(np.asarray(y_score))

        fold_state.n_train_sizes.append(int(np.asarray(Xtr).shape[0]))
        fold_state.n_test_sizes.append(int(np.asarray(Xte).shape[0]))

    if last_model is None:
        raise ValueError("No folds produced by splitter; cannot train ensemble.")

    return last_model
