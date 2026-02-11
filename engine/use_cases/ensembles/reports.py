from __future__ import annotations

from engine.contracts.ensemble_configs import (
    AdaBoostEnsembleConfig,
    BaggingEnsembleConfig,
    VotingEnsembleConfig,
    XGBoostEnsembleConfig,
)
from engine.reporting.ensembles.adaboost_ensemble_reporting import (
    AdaBoostEnsembleReportAccumulator,
    AdaBoostEnsembleRegressorReportAccumulator,
)
from engine.reporting.ensembles.bagging_ensemble_reporting import (
    BaggingEnsembleReportAccumulator,
    BaggingEnsembleRegressorReportAccumulator,
)
from engine.reporting.ensembles.voting_ensemble_reporting import (
    VotingEnsembleReportAccumulator,
    VotingEnsembleRegressorReportAccumulator,
)
from engine.reporting.ensembles.xgboost_ensemble_reporting import XGBoostEnsembleReportAccumulator

from .types import ReportState


def init_report_state(cfg) -> ReportState:
    """Initialize report accumulator state based on ensemble config type."""

    rs = ReportState()

    rs.is_voting = isinstance(cfg.ensemble, VotingEnsembleConfig)
    rs.is_bagging = isinstance(cfg.ensemble, BaggingEnsembleConfig)
    rs.is_adaboost = isinstance(cfg.ensemble, AdaBoostEnsembleConfig)
    rs.is_xgboost = isinstance(cfg.ensemble, XGBoostEnsembleConfig)

    # Accumulators start as None and are created by update_*_report functions.
    rs.voting_cls_acc = None
    rs.voting_reg_acc = None
    rs.bagging_cls_acc = None
    rs.bagging_reg_acc = None
    rs.adaboost_cls_acc = None
    rs.adaboost_reg_acc = None
    rs.xgb_acc = None

    return rs


def finalize_ensemble_report(rs: ReportState):
    """Finalize whichever report accumulator was active."""

    if rs.voting_cls_acc is not None:
        return rs.voting_cls_acc.finalize()
    if rs.voting_reg_acc is not None:
        return rs.voting_reg_acc.finalize()

    if rs.bagging_cls_acc is not None:
        return rs.bagging_cls_acc.finalize()
    if rs.bagging_reg_acc is not None:
        return rs.bagging_reg_acc.finalize()

    if rs.adaboost_cls_acc is not None:
        return rs.adaboost_cls_acc.finalize()
    if rs.adaboost_reg_acc is not None:
        return rs.adaboost_reg_acc.finalize()

    if rs.xgb_acc is not None:
        return rs.xgb_acc.finalize()

    return None
