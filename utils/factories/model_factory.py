# utils/factories/model_factory.py
from __future__ import annotations
from typing import Any

from utils.configs.configs import ModelConfig
from utils.strategies.interfaces import ModelBuilder
from utils.strategies.models import LogRegBuilder


def make_model(cfg: ModelConfig) -> ModelBuilder:
    """
    Create a model builder strategy from config.
    Today: only 'logreg'. Add branches here for future models.
    """
    algo = (cfg.algo or "logreg").lower()

    if algo == "logreg":
        return LogRegBuilder(cfg=cfg)

    raise ValueError(f"Unknown model algo: {cfg.algo}")
