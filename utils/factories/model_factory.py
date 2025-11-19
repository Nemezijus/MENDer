from shared_schemas.model_configs import ModelModel
from utils.strategies.interfaces import ModelBuilder
from utils.strategies.models import (
    LogRegBuilder, SVMBuilder,
    DecisionTreeBuilder, RandomForestBuilder, KNNBuilder,
)

def make_model(cfg: ModelModel) -> ModelBuilder:
    algo = (cfg.algo or "logreg").lower()
    if algo == "logreg":
        return LogRegBuilder(cfg=cfg)
    if algo == "svm":
        return SVMBuilder(cfg=cfg)
    if algo == "tree":
        return DecisionTreeBuilder(cfg=cfg)
    if algo == "forest":
        return RandomForestBuilder(cfg=cfg)
    if algo == "knn":
        return KNNBuilder(cfg=cfg)
    raise ValueError(f"Unknown model algo: {cfg.algo}")
