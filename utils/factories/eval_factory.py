# utils/factories/eval_factory.py
from __future__ import annotations
from typing import Literal

from utils.configs.configs import EvalConfig
from utils.strategies.interfaces import Evaluator
from utils.strategies.evaluators import SklearnEvaluator

def make_evaluator(
    cfg: EvalConfig,
    *,
    kind: Literal["classification","regression"] = "classification",
) -> Evaluator:
    """
    Create an evaluator strategy from config.
    """
    return SklearnEvaluator(cfg=cfg, kind=kind)
