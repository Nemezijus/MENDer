from __future__ import annotations
from sklearn.pipeline import Pipeline

from shared_schemas.run_config import RunConfig
from shared_schemas.scale_configs import ScaleModel
from shared_schemas.feature_configs import FeaturesModel
from shared_schemas.eval_configs import EvalModel
from shared_schemas.model_configs import ModelConfig
from utils.permutations.rng import RngManager

from utils.factories.scale_factory import make_scaler
from utils.factories.feature_factory import make_features
from utils.factories.model_factory import make_model

def make_pipeline(cfg: RunConfig, rngm: RngManager, *, stream: str = "real") -> Pipeline:
    features_seed = rngm.child_seed(f"{stream}/features")

    scaler_strategy  = make_scaler(cfg.scale)
    feature_strategy = make_features(cfg.features, seed=features_seed, model_cfg=cfg.model, eval_cfg=cfg.eval)
    model_seed = rngm.child_seed(f"{stream}/model")
    model_builder = make_model(cfg.model, seed=model_seed)

    return Pipeline(steps=[
        ("scale", scaler_strategy.make_transformer()),
        ("feat",  feature_strategy.make_transformer()),
        ("clf",   model_builder.make_estimator()),
    ])

def make_preproc_pipeline(cfg: RunConfig, rngm: RngManager, *, stream: str = "real") -> Pipeline:
    features_seed = rngm.child_seed(f"{stream}/features")

    scaler_strategy  = make_scaler(cfg.scale)
    feature_strategy = make_features(cfg.features, seed=features_seed, model_cfg=cfg.model, eval_cfg=cfg.eval)

    return Pipeline(steps=[
        ("scale", scaler_strategy.make_transformer()),
        ("feat",  feature_strategy.make_transformer()),
    ])


def make_pipeline_for_model_cfg(
    scale: ScaleModel,
    features: FeaturesModel,
    model_cfg: ModelConfig,
    eval_cfg: EvalModel,
    rngm: RngManager,
    *,
    stream: str = "real",
) -> Pipeline:
    """Build a (scale -> features -> estimator) pipeline for an explicit ModelConfig.

    This is useful for ensembles, where each base estimator may have a different
    ModelConfig but should share the same preprocessing config.
    """
    features_seed = rngm.child_seed(f"{stream}/features")

    scaler_strategy  = make_scaler(scale)
    feature_strategy = make_features(features, seed=features_seed, model_cfg=model_cfg, eval_cfg=eval_cfg)
    model_seed = rngm.child_seed(f"{stream}/model")
    model_builder = make_model(model_cfg, seed=model_seed)

    return Pipeline(steps=[
        ("scale", scaler_strategy.make_transformer()),
        ("feat",  feature_strategy.make_transformer()),
        ("clf",   model_builder.make_estimator()),
    ])

def make_preproc_pipeline_for_model_cfg(
    scale: ScaleModel,
    features: FeaturesModel,
    model_cfg: ModelConfig | None,
    eval_cfg: EvalModel,
    rngm: RngManager,
    *,
    stream: str = "real",
) -> Pipeline:
    """Build a preprocessing-only pipeline: (scale -> features).

    Used by boosting-style ensembles (e.g. AdaBoost) where the ensemble needs to
    pass `sample_weight` into the base estimator. If the base estimator were a
    full sklearn Pipeline, AdaBoost would pass `sample_weight` to Pipeline.fit()
    which is not supported, causing:
        "Pipeline doesn't support sample_weight".

    `model_cfg` is provided so feature strategies (e.g. SFS) can configure themselves
    consistently with the estimator family. For default cases it may be None; if a
    feature strategy requires `model_cfg`, it should raise a clear error.
    """
    features_seed = rngm.child_seed(f"{stream}/features")

    scaler_strategy  = make_scaler(scale)
    feature_strategy = make_features(features, seed=features_seed, model_cfg=model_cfg, eval_cfg=eval_cfg)

    return Pipeline(steps=[
        ("scale", scaler_strategy.make_transformer()),
        ("feat",  feature_strategy.make_transformer()),
    ])