from shared_schemas.model_configs import (
  ModelConfig, LogRegConfig, SVMConfig, TreeConfig, ForestConfig, KNNConfig
)
from utils.strategies.models import (
  LogRegBuilder, SVMBuilder, DecisionTreeBuilder, RandomForestBuilder, KNNBuilder
)

def make_model(cfg: ModelConfig):
    if isinstance(cfg, LogRegConfig):  return LogRegBuilder(cfg)
    if isinstance(cfg, SVMConfig):     return SVMBuilder(cfg)
    if isinstance(cfg, TreeConfig):    return DecisionTreeBuilder(cfg)
    if isinstance(cfg, ForestConfig):  return RandomForestBuilder(cfg)
    if isinstance(cfg, KNNConfig):     return KNNBuilder(cfg)
    raise ValueError(f"Unsupported algo: {getattr(cfg, 'algo', None)}")