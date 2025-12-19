from shared_schemas.model_configs import (
  ModelConfig, LogRegConfig, SVMConfig, TreeConfig, ForestConfig, KNNConfig,
  LinearRegConfig
)
from utils.strategies.models import (
  LogRegBuilder, SVMBuilder, DecisionTreeBuilder, RandomForestBuilder, KNNBuilder,
  LinRegBuilder
)

def make_model(cfg: ModelConfig, *, seed: int | None = None):
    if isinstance(cfg, LogRegConfig):         return LogRegBuilder(cfg, seed=seed)
    if isinstance(cfg, SVMConfig):            return SVMBuilder(cfg, seed=seed)
    if isinstance(cfg, TreeConfig):           return DecisionTreeBuilder(cfg, seed=seed)
    if isinstance(cfg, ForestConfig):         return RandomForestBuilder(cfg, seed=seed)
    if isinstance(cfg, KNNConfig):            return KNNBuilder(cfg)
    if isinstance(cfg, LinearRegConfig):      return LinRegBuilder(cfg)
    raise ValueError(f"Unsupported algo: {getattr(cfg, 'algo', None)}")