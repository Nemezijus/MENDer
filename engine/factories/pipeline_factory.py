from __future__ import annotations

from sklearn.pipeline import Pipeline

from engine.contracts.eval_configs import EvalModel
from engine.contracts.feature_configs import FeaturesModel
from engine.contracts.model_configs import ModelConfig
from engine.contracts.run_config import RunConfig
from engine.contracts.scale_configs import ScaleModel
from engine.contracts.unsupervised_configs import UnsupervisedRunConfig
from engine.runtime.random.rng import RngManager
from engine.types.sklearn import SkEstimator, SkPipeline

from engine.factories.feature_factory import make_features
from engine.factories.model_factory import make_model
from engine.factories.scale_factory import make_scaler


def _make_pipeline(
    *,
    scale: ScaleModel,
    features: FeaturesModel,
    rngm: RngManager,
    include_estimator: bool,
    unsupervised: bool,
    model_cfg: ModelConfig | None = None,
    eval_cfg: EvalModel | None = None,
    model_cfg_override: ModelConfig | None = None,
    stream: str = "real",
) -> SkPipeline:
    """Build a (scale -> features -> [estimator]) sklearn Pipeline.

    Parameters
    ----------
    include_estimator
        If True, append the final estimator step (named "clf").
    unsupervised
        If True, feature strategy selection will be called with eval_cfg=None.
    model_cfg
        The default model config. May be None for preprocessing-only pipelines that do not
        need an estimator (e.g. boosting ensembles that must pass sample_weight into the
        base estimator).
    eval_cfg
        Passed into feature strategy selection for supervised runs.
    model_cfg_override
        If provided, overrides `model_cfg` when building the estimator and/or configuring
        feature strategies. This is primarily used by ensembles where base estimators may
        vary but preprocessing remains shared.

    Notes
    -----
    - The step name "clf" is used consistently across the project as the final estimator
      step, even for unsupervised estimators.
    """
    features_seed = rngm.child_seed(f"{stream}/features")

    scaler_strategy = make_scaler(scale)

    effective_model_cfg = model_cfg_override if model_cfg_override is not None else model_cfg
    eval_cfg_for_features: EvalModel | None = None if unsupervised else eval_cfg
    feature_strategy = make_features(
        features,
        seed=features_seed,
        model_cfg=effective_model_cfg,
        eval_cfg=eval_cfg_for_features,
    )

    steps: list[tuple[str, object]] = [
        ("scale", scaler_strategy.make_transformer()),
        ("feat", feature_strategy.make_transformer()),
    ]

    if include_estimator:
        if effective_model_cfg is None:
            raise ValueError("model_cfg is required when include_estimator=True")
        model_seed = rngm.child_seed(f"{stream}/model")
        model_builder = make_model(effective_model_cfg, seed=model_seed)
        est: SkEstimator = model_builder.make_estimator()
        steps.append(("clf", est))

    return Pipeline(steps=steps)


def make_pipeline(cfg: RunConfig, rngm: RngManager, *, stream: str = "real") -> SkPipeline:
    return _make_pipeline(
        scale=cfg.scale,
        features=cfg.features,
        model_cfg=cfg.model,
        eval_cfg=cfg.eval,
        rngm=rngm,
        include_estimator=True,
        unsupervised=False,
        stream=stream,
    )


def make_preproc_pipeline(cfg: RunConfig, rngm: RngManager, *, stream: str = "real") -> SkPipeline:
    return _make_pipeline(
        scale=cfg.scale,
        features=cfg.features,
        model_cfg=cfg.model,
        eval_cfg=cfg.eval,
        rngm=rngm,
        include_estimator=False,
        unsupervised=False,
        stream=stream,
    )


def make_unsupervised_pipeline(cfg: UnsupervisedRunConfig, rngm: RngManager, *, stream: str = "real") -> SkPipeline:
    """Build a (scale -> features -> estimator) pipeline for unsupervised learning."""
    return _make_pipeline(
        scale=cfg.scale,
        features=cfg.features,
        model_cfg=cfg.model,
        rngm=rngm,
        include_estimator=True,
        unsupervised=True,
        stream=stream,
    )


def make_unsupervised_preproc_pipeline(
    cfg: UnsupervisedRunConfig, rngm: RngManager, *, stream: str = "real"
) -> SkPipeline:
    """Build a preprocessing-only pipeline for unsupervised learning: (scale -> features)."""
    # Unsupervised feature strategies should not depend on supervised evaluation config.
    # If a caller selects a feature method that requires `eval_cfg` (e.g. SFS),
    # make_features will raise a clear error.
    return _make_pipeline(
        scale=cfg.scale,
        features=cfg.features,
        model_cfg=cfg.model,
        rngm=rngm,
        include_estimator=False,
        unsupervised=True,
        stream=stream,
    )


def make_pipeline_for_model_cfg(
    scale: ScaleModel,
    features: FeaturesModel,
    model_cfg: ModelConfig,
    eval_cfg: EvalModel,
    rngm: RngManager,
    *,
    stream: str = "real",
) -> SkPipeline:
    """Build a (scale -> features -> estimator) pipeline for an explicit ModelConfig.

    This is useful for ensembles, where each base estimator may have a different
    ModelConfig but should share the same preprocessing config.
    """
    return _make_pipeline(
        scale=scale,
        features=features,
        model_cfg=model_cfg,
        eval_cfg=eval_cfg,
        rngm=rngm,
        include_estimator=True,
        unsupervised=False,
        stream=stream,
    )


def make_preproc_pipeline_for_model_cfg(
    scale: ScaleModel,
    features: FeaturesModel,
    model_cfg: ModelConfig | None,
    eval_cfg: EvalModel,
    rngm: RngManager,
    *,
    stream: str = "real",
) -> SkPipeline:
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
    return _make_pipeline(
        scale=scale,
        features=features,
        model_cfg=model_cfg,
        eval_cfg=eval_cfg,
        rngm=rngm,
        include_estimator=False,
        unsupervised=False,
        stream=stream,
    )
