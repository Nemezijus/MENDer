from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import inspect

from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering, MeanShift, Birch
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from engine.contracts.model_configs import (
    KMeansConfig,
    DBSCANConfig,
    SpectralClusteringConfig,
    AgglomerativeClusteringConfig,
    GaussianMixtureConfig,
    BayesianGaussianMixtureConfig,
    MeanShiftConfig,
    BirchConfig,
)
from engine.components.interfaces import ModelBuilder
from engine.types.sklearn import SkClusterer

from .common import _filtered_kwargs, _maybe_set_random_state


@dataclass
class KMeansBuilder(ModelBuilder):
    cfg: KMeansConfig
    seed: Optional[int] = None

    def make_estimator(self) -> SkClusterer:
        kw = _filtered_kwargs(KMeans, self.cfg)
        _maybe_set_random_state(KMeans, kw, self.seed)
        return KMeans(**kw)

    def build(self) -> SkClusterer:
        return self.make_estimator()


@dataclass
class DBSCANBuilder(ModelBuilder):
    cfg: DBSCANConfig

    def make_estimator(self) -> SkClusterer:
        kw = _filtered_kwargs(DBSCAN, self.cfg)
        return DBSCAN(**kw)

    def build(self) -> SkClusterer:
        return self.make_estimator()


@dataclass
class SpectralClusteringBuilder(ModelBuilder):
    cfg: SpectralClusteringConfig
    seed: Optional[int] = None

    def make_estimator(self) -> SkClusterer:
        kw = _filtered_kwargs(SpectralClustering, self.cfg)
        _maybe_set_random_state(SpectralClustering, kw, self.seed)
        return SpectralClustering(**kw)

    def build(self) -> SkClusterer:
        return self.make_estimator()


@dataclass
class AgglomerativeClusteringBuilder(ModelBuilder):
    cfg: AgglomerativeClusteringConfig

    def make_estimator(self) -> SkClusterer:
        # sklearn changed affinity -> metric; support both without renaming cfg fields.
        raw = self.cfg.model_dump(exclude={"algo"}, exclude_none=True)
        sig = inspect.signature(AgglomerativeClustering)
        allowed = set(sig.parameters.keys())

        kw: Dict[str, Any] = {k: v for k, v in raw.items() if k in allowed}

        if "metric" not in allowed and "affinity" in allowed and "metric" in raw:
            kw["affinity"] = raw["metric"]

        return AgglomerativeClustering(**kw)

    def build(self) -> SkClusterer:
        return self.make_estimator()


@dataclass
class GaussianMixtureBuilder(ModelBuilder):
    cfg: GaussianMixtureConfig
    seed: Optional[int] = None

    def make_estimator(self) -> SkClusterer:
        kw = _filtered_kwargs(GaussianMixture, self.cfg)
        _maybe_set_random_state(GaussianMixture, kw, self.seed)
        return GaussianMixture(**kw)

    def build(self) -> SkClusterer:
        return self.make_estimator()


@dataclass
class BayesianGaussianMixtureBuilder(ModelBuilder):
    cfg: BayesianGaussianMixtureConfig
    seed: Optional[int] = None

    def make_estimator(self) -> SkClusterer:
        kw = _filtered_kwargs(BayesianGaussianMixture, self.cfg)
        _maybe_set_random_state(BayesianGaussianMixture, kw, self.seed)
        return BayesianGaussianMixture(**kw)

    def build(self) -> SkClusterer:
        return self.make_estimator()


@dataclass
class MeanShiftBuilder(ModelBuilder):
    cfg: MeanShiftConfig

    def make_estimator(self) -> SkClusterer:
        kw = _filtered_kwargs(MeanShift, self.cfg)
        return MeanShift(**kw)

    def build(self) -> SkClusterer:
        return self.make_estimator()


@dataclass
class BirchBuilder(ModelBuilder):
    cfg: BirchConfig

    def make_estimator(self) -> SkClusterer:
        kw = _filtered_kwargs(Birch, self.cfg)
        return Birch(**kw)

    def build(self) -> SkClusterer:
        return self.make_estimator()
