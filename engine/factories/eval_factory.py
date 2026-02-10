from __future__ import annotations
from typing import Literal

from engine.contracts.eval_configs import EvalModel
from engine.contracts.unsupervised_configs import UnsupervisedEvalModel
from engine.components.interfaces import Evaluator, UnsupervisedEvaluator
from engine.components.evaluation.evaluators import SklearnEvaluator, SklearnUnsupervisedEvaluator

def make_evaluator(
    cfg: EvalModel,
    *,
    kind: Literal["classification","regression"] = "classification",
) -> Evaluator:
    """
    Create an evaluator strategy from config.
    """
    return SklearnEvaluator(cfg=cfg, kind=kind)


def make_unsupervised_evaluator(cfg: UnsupervisedEvalModel) -> UnsupervisedEvaluator:
    """Create an unsupervised evaluator strategy from config."""
    return SklearnUnsupervisedEvaluator(cfg=cfg)
