# utils/factories/model_factory.py
from shared_schemas.model_configs import (
  ModelConfig,
  LogRegConfig, SVMConfig, TreeConfig, ForestConfig, KNNConfig,
  GaussianNBConfig, RidgeClassifierConfig, SGDClassifierConfig,
  ExtraTreesConfig, HistGradientBoostingConfig,
  LinearRegConfig,

  RidgeRegressorConfig,
  RidgeCVRegressorConfig,
  ElasticNetRegressorConfig,
  ElasticNetCVRegressorConfig,
  LassoRegressorConfig,
  LassoCVRegressorConfig,
  BayesianRidgeRegressorConfig,
  SVRRegressorConfig,
  LinearSVRRegressorConfig,
  KNNRegressorConfig,
  DecisionTreeRegressorConfig,
  RandomForestRegressorConfig,
)
from utils.strategies.models import (
  LogRegBuilder, SVMBuilder, DecisionTreeBuilder, RandomForestBuilder, KNNBuilder,
  GaussianNBBuilder, RidgeClassifierBuilder, SGDClassifierBuilder,
  ExtraTreesBuilder, HistGradientBoostingBuilder,
  LinRegBuilder,

  RidgeRegressorBuilder,
  RidgeCVRegressorBuilder,
  ElasticNetRegressorBuilder,
  ElasticNetCVRegressorBuilder,
  LassoRegressorBuilder,
  LassoCVRegressorBuilder,
  BayesianRidgeRegressorBuilder,
  SVRRegressorBuilder,
  LinearSVRRegressorBuilder,
  KNNRegressorBuilder,
  DecisionTreeRegressorBuilder,
  RandomForestRegressorBuilder,
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

    # ---------------- regressors ----------------
    if isinstance(cfg, LinearRegConfig):      return LinRegBuilder(cfg)
    if isinstance(cfg, RidgeRegressorConfig): return RidgeRegressorBuilder(cfg, seed=seed)
    if isinstance(cfg, RidgeCVRegressorConfig): return RidgeCVRegressorBuilder(cfg)
    if isinstance(cfg, ElasticNetRegressorConfig): return ElasticNetRegressorBuilder(cfg, seed=seed)
    if isinstance(cfg, ElasticNetCVRegressorConfig): return ElasticNetCVRegressorBuilder(cfg, seed=seed)
    if isinstance(cfg, LassoRegressorConfig): return LassoRegressorBuilder(cfg, seed=seed)
    if isinstance(cfg, LassoCVRegressorConfig): return LassoCVRegressorBuilder(cfg, seed=seed)
    if isinstance(cfg, BayesianRidgeRegressorConfig): return BayesianRidgeRegressorBuilder(cfg)
    if isinstance(cfg, SVRRegressorConfig):   return SVRRegressorBuilder(cfg)
    if isinstance(cfg, LinearSVRRegressorConfig): return LinearSVRRegressorBuilder(cfg, seed=seed)
    if isinstance(cfg, KNNRegressorConfig):   return KNNRegressorBuilder(cfg)
    if isinstance(cfg, DecisionTreeRegressorConfig): return DecisionTreeRegressorBuilder(cfg, seed=seed)
    if isinstance(cfg, RandomForestRegressorConfig): return RandomForestRegressorBuilder(cfg, seed=seed)
    raise ValueError(f"Unsupported algo: {getattr(cfg, 'algo', None)}")
