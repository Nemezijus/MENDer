# utils/factories/model_factory.py
from utils.configs.configs import ModelConfig
from utils.strategies.interfaces import ModelBuilder
from utils.strategies.models import (
    LogRegBuilder, SVMBuilder,
    DecisionTreeBuilder, RandomForestBuilder,
)

def make_model(cfg: ModelConfig) -> ModelBuilder:
    algo = (cfg.algo or "logreg").lower()
    if algo == "logreg":
        return LogRegBuilder(cfg=cfg)
    if algo == "svm":
        return SVMBuilder(cfg=cfg)
    if algo == "tree":
        return DecisionTreeBuilder(cfg=cfg)
    if algo == "forest":
        return RandomForestBuilder(cfg=cfg)
    raise ValueError(f"Unknown model algo: {cfg.algo}")
