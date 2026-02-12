import numpy as np
from typing import Any, Tuple

from engine.contracts.ensemble_run_config import EnsembleRunConfig
from sklearn.ensemble import VotingClassifier, VotingRegressor

from engine.reporting.ensembles.voting import (
    VotingEnsembleReportAccumulator,
    VotingEnsembleRegressorReportAccumulator,
)

from ..helpers import _unwrap_final_estimator
from ..common import attach_report_error


def update_voting_report(
    *,
    cfg: EnsembleRunConfig,
    eval_kind: str,
    model: Any,
    Xtr: Any,
    Xte: Any,
    ytr: Any,
    yte: Any,
    y_pred: Any,
    fold_id: int,
    evaluator: Any,
    voting_cls_acc: VotingEnsembleReportAccumulator | None,
    voting_reg_acc: VotingEnsembleRegressorReportAccumulator | None,
) -> Tuple[VotingEnsembleReportAccumulator | None, VotingEnsembleRegressorReportAccumulator | None]:
    """Update Voting ensemble report accumulators for the current fold (best-effort, never raises)."""
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
    except Exception as e:
        if voting_cls_acc is not None:
            attach_report_error(voting_cls_acc, where="ensembles.reports.voting", exc=e, context={"fold_id": fold_id, "eval_kind": eval_kind})
        if voting_reg_acc is not None:
            attach_report_error(voting_reg_acc, where="ensembles.reports.voting", exc=e, context={"fold_id": fold_id, "eval_kind": eval_kind})
        return voting_cls_acc, voting_reg_acc

    return voting_cls_acc, voting_reg_acc
