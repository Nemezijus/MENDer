from __future__ import annotations
from typing import Literal

from shared_schemas.eval_configs import EvalModel
from utils.strategies.interfaces import Evaluator
from utils.strategies.evaluators import SklearnEvaluator

def make_evaluator(
    cfg: EvalModel,
    *,
    kind: Literal["classification","regression"] = "classification",
) -> Evaluator:
    """
    Create an evaluator strategy from config.
    """
    return SklearnEvaluator(cfg=cfg, kind=kind)
