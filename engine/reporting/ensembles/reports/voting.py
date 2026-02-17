import numpy as np
from typing import Any, Tuple

from engine.contracts.ensemble_run_config import EnsembleRunConfig
from sklearn.ensemble import VotingClassifier, VotingRegressor

from engine.reporting.ensembles.voting import (
    VotingEnsembleReportAccumulator,
    VotingEnsembleRegressorReportAccumulator,
)

from .extract import collect_voting_base_preds_and_scores, resolve_estimator_and_X
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
        final_est, Xte_use = resolve_estimator_and_X(model, Xte)

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

            base_preds, base_scores = collect_voting_base_preds_and_scores(
                voting_estimator=final_est,
                X=Xte_use,
                y_true=yte,
                evaluator=evaluator,
                is_classification=True,
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

            base_preds, base_scores = collect_voting_base_preds_and_scores(
                voting_estimator=final_est,
                X=Xte_use,
                y_true=yte,
                evaluator=evaluator,
                is_classification=False,
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
