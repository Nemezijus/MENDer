"""Built-in model builder registrations.

This module is imported for side-effects by :mod:`engine.registries.models`.

Goal: avoid factory if/else sprawl.

To add a new model:
    1) create a new builder class/factory
    2) register it via register_model_builder

In later segments we will migrate builders out of utils into engine, but
registries already provide the stable extension surface.
"""

from __future__ import annotations

from typing import Optional, Type

from engine.registries.models import ModelBuilderFactory, register_model_builder

from shared_schemas.model_configs import (
    LogRegConfig,
    SVMConfig,
    TreeConfig,
    ForestConfig,
    KNNConfig,
    GaussianNBConfig,
    RidgeClassifierConfig,
    SGDClassifierConfig,
    ExtraTreesConfig,
    HistGradientBoostingConfig,
    # regressors
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
    # unsupervised
    KMeansConfig,
    DBSCANConfig,
    SpectralClusteringConfig,
    AgglomerativeClusteringConfig,
    GaussianMixtureConfig,
    BayesianGaussianMixtureConfig,
    MeanShiftConfig,
    BirchConfig,
)

from utils.strategies.models import (
    LogRegBuilder,
    SVMBuilder,
    DecisionTreeBuilder,
    RandomForestBuilder,
    KNNBuilder,
    GaussianNBBuilder,
    RidgeClassifierBuilder,
    SGDClassifierBuilder,
    ExtraTreesBuilder,
    HistGradientBoostingBuilder,
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
    KMeansBuilder,
    DBSCANBuilder,
    SpectralClusteringBuilder,
    AgglomerativeClusteringBuilder,
    GaussianMixtureBuilder,
    BayesianGaussianMixtureBuilder,
    MeanShiftBuilder,
    BirchBuilder,
)


def _default_algo(cfg_type: Type) -> Optional[str]:
    """Best-effort extraction of a config's default algo key."""
    try:
        inst = cfg_type()  # type: ignore[call-arg]
        return str(getattr(inst, "algo", None))
    except Exception:
        return None


def _factory(builder_type: Type, cfg_type: Type) -> ModelBuilderFactory:
    """Wrap a builder class into the common (cfg, seed) factory signature."""

    def make(cfg, seed: Optional[int]):
        try:
            return builder_type(cfg, seed=seed)
        except TypeError:
            # builder doesn't accept seed
            return builder_type(cfg)

    algo = _default_algo(cfg_type)
    # Register by config type; also by algo if available.
    register_model_builder(cfg_type, algo=algo)(make)
    return make


# ---------------- classification ----------------
_factory(LogRegBuilder, LogRegConfig)
_factory(SVMBuilder, SVMConfig)
_factory(DecisionTreeBuilder, TreeConfig)
_factory(RandomForestBuilder, ForestConfig)
_factory(KNNBuilder, KNNConfig)
_factory(GaussianNBBuilder, GaussianNBConfig)
_factory(RidgeClassifierBuilder, RidgeClassifierConfig)
_factory(SGDClassifierBuilder, SGDClassifierConfig)
_factory(ExtraTreesBuilder, ExtraTreesConfig)
_factory(HistGradientBoostingBuilder, HistGradientBoostingConfig)

# ---------------- regression ----------------
_factory(LinRegBuilder, LinearRegConfig)
_factory(RidgeRegressorBuilder, RidgeRegressorConfig)
_factory(RidgeCVRegressorBuilder, RidgeCVRegressorConfig)
_factory(ElasticNetRegressorBuilder, ElasticNetRegressorConfig)
_factory(ElasticNetCVRegressorBuilder, ElasticNetCVRegressorConfig)
_factory(LassoRegressorBuilder, LassoRegressorConfig)
_factory(LassoCVRegressorBuilder, LassoCVRegressorConfig)
_factory(BayesianRidgeRegressorBuilder, BayesianRidgeRegressorConfig)
_factory(SVRRegressorBuilder, SVRRegressorConfig)
_factory(LinearSVRRegressorBuilder, LinearSVRRegressorConfig)
_factory(KNNRegressorBuilder, KNNRegressorConfig)
_factory(DecisionTreeRegressorBuilder, DecisionTreeRegressorConfig)
_factory(RandomForestRegressorBuilder, RandomForestRegressorConfig)

# ---------------- unsupervised ----------------
_factory(KMeansBuilder, KMeansConfig)
_factory(DBSCANBuilder, DBSCANConfig)
_factory(SpectralClusteringBuilder, SpectralClusteringConfig)
_factory(AgglomerativeClusteringBuilder, AgglomerativeClusteringConfig)
_factory(GaussianMixtureBuilder, GaussianMixtureConfig)
_factory(BayesianGaussianMixtureBuilder, BayesianGaussianMixtureConfig)
_factory(MeanShiftBuilder, MeanShiftConfig)
_factory(BirchBuilder, BirchConfig)
