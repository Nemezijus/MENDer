from __future__ import annotations
from sklearn.pipeline import Pipeline

from shared_schemas.run_config import RunConfig
from utils.permutations.rng import RngManager

from utils.factories.scale_factory import make_scaler
from utils.factories.feature_factory import make_features
from utils.factories.model_factory import make_model

def make_pipeline(cfg: RunConfig, rngm: RngManager, *, stream: str = "real") -> Pipeline:
    features_seed = rngm.child_seed(f"{stream}/features")

    scaler_strategy  = make_scaler(cfg.scale)
    feature_strategy = make_features(cfg.features, seed=features_seed, model_cfg=cfg.model, eval_cfg=cfg.eval)
    model_builder    = make_model(cfg.model)

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
