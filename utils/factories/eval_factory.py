from __future__ import annotations
from typing import Literal

from shared_schemas.eval_configs import EvalModel
from shared_schemas.unsupervised_configs import UnsupervisedEvalModel
from utils.strategies.interfaces import Evaluator, UnsupervisedEvaluator
from utils.strategies.evaluators import SklearnEvaluator, SklearnUnsupervisedEvaluator

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
