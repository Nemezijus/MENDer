# utils/factories/model_factory.py
from shared_schemas.model_configs import (
  ModelConfig,
  LogRegConfig, SVMConfig, TreeConfig, ForestConfig, KNNConfig,
  GaussianNBConfig, RidgeClassifierConfig, SGDClassifierConfig,
  ExtraTreesConfig, HistGradientBoostingConfig,
  LinearRegConfig,
)
from utils.strategies.models import (
  LogRegBuilder, SVMBuilder, DecisionTreeBuilder, RandomForestBuilder, KNNBuilder,
  GaussianNBBuilder, RidgeClassifierBuilder, SGDClassifierBuilder,
  ExtraTreesBuilder, HistGradientBoostingBuilder,
  LinRegBuilder,
)

def make_model(cfg: ModelConfig, *, seed: int | None = None):
    if isinstance(cfg, LogRegConfig):         return LogRegBuilder(cfg, seed=seed)
    if isinstance(cfg, SVMConfig):            return SVMBuilder(cfg, seed=seed)
    if isinstance(cfg, TreeConfig):           return DecisionTreeBuilder(cfg, seed=seed)
    if isinstance(cfg, ForestConfig):         return RandomForestBuilder(cfg, seed=seed)
    if isinstance(cfg, KNNConfig):            return KNNBuilder(cfg)
    if isinstance(cfg, GaussianNBConfig):     return GaussianNBBuilder(cfg)
    if isinstance(cfg, RidgeClassifierConfig): return RidgeClassifierBuilder(cfg, seed=seed)
    if isinstance(cfg, SGDClassifierConfig):  return SGDClassifierBuilder(cfg, seed=seed)
    if isinstance(cfg, ExtraTreesConfig):     return ExtraTreesBuilder(cfg, seed=seed)
    if isinstance(cfg, HistGradientBoostingConfig): return HistGradientBoostingBuilder(cfg, seed=seed)
    if isinstance(cfg, LinearRegConfig):      return LinRegBuilder(cfg)
    raise ValueError(f"Unsupported algo: {getattr(cfg, 'algo', None)}")
