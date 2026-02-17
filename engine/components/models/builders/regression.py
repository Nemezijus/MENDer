from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sklearn.linear_model import (LinearRegression, Ridge, RidgeCV, ElasticNet, ElasticNetCV, Lasso, LassoCV, BayesianRidge)
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from engine.contracts.model_configs import (
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
from engine.components.interfaces import ModelBuilder
from engine.types.sklearn import SkRegressor

from .common import _filtered_kwargs, _maybe_set_random_state



@dataclass
class LinRegBuilder(ModelBuilder):
    cfg: LinearRegConfig

    def make_estimator(self) -> SkRegressor:
        kw = _filtered_kwargs(LinearRegression, self.cfg)
        return LinearRegression(**kw)

    def build(self) -> SkRegressor:
        return self.make_estimator()


@dataclass
class RidgeRegressorBuilder(ModelBuilder):
    cfg: RidgeRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> SkRegressor:
        kw = _filtered_kwargs(Ridge, self.cfg)
        _maybe_set_random_state(Ridge, kw, self.seed)
        return Ridge(**kw)

    def build(self) -> SkRegressor:
        return self.make_estimator()


@dataclass
class RidgeCVRegressorBuilder(ModelBuilder):
    cfg: RidgeCVRegressorConfig

    def make_estimator(self) -> SkRegressor:
        kw = _filtered_kwargs(RidgeCV, self.cfg)
        return RidgeCV(**kw)

    def build(self) -> SkRegressor:
        return self.make_estimator()


@dataclass
class ElasticNetRegressorBuilder(ModelBuilder):
    cfg: ElasticNetRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> SkRegressor:
        kw = _filtered_kwargs(ElasticNet, self.cfg)
        _maybe_set_random_state(ElasticNet, kw, self.seed)
        return ElasticNet(**kw)

    def build(self) -> SkRegressor:
        return self.make_estimator()


@dataclass
class ElasticNetCVRegressorBuilder(ModelBuilder):
    cfg: ElasticNetCVRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> SkRegressor:
        kw = _filtered_kwargs(ElasticNetCV, self.cfg)
        _maybe_set_random_state(ElasticNetCV, kw, self.seed)
        return ElasticNetCV(**kw)

    def build(self) -> SkRegressor:
        return self.make_estimator()


@dataclass
class LassoRegressorBuilder(ModelBuilder):
    cfg: LassoRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> SkRegressor:
        kw = _filtered_kwargs(Lasso, self.cfg)
        _maybe_set_random_state(Lasso, kw, self.seed)
        return Lasso(**kw)

    def build(self) -> SkRegressor:
        return self.make_estimator()


@dataclass
class LassoCVRegressorBuilder(ModelBuilder):
    cfg: LassoCVRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> SkRegressor:
        kw = _filtered_kwargs(LassoCV, self.cfg)
        _maybe_set_random_state(LassoCV, kw, self.seed)
        return LassoCV(**kw)

    def build(self) -> SkRegressor:
        return self.make_estimator()


@dataclass
class BayesianRidgeRegressorBuilder(ModelBuilder):
    cfg: BayesianRidgeRegressorConfig

    def make_estimator(self) -> SkRegressor:
        kw = _filtered_kwargs(BayesianRidge, self.cfg)
        return BayesianRidge(**kw)

    def build(self) -> SkRegressor:
        return self.make_estimator()


@dataclass
class SVRRegressorBuilder(ModelBuilder):
    cfg: SVRRegressorConfig

    def make_estimator(self) -> SkRegressor:
        kw = _filtered_kwargs(SVR, self.cfg)
        return SVR(**kw)

    def build(self) -> SkRegressor:
        return self.make_estimator()


@dataclass
class LinearSVRRegressorBuilder(ModelBuilder):
    cfg: LinearSVRRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> SkRegressor:
        kw = _filtered_kwargs(LinearSVR, self.cfg)
        _maybe_set_random_state(LinearSVR, kw, self.seed)
        return LinearSVR(**kw)

    def build(self) -> SkRegressor:
        return self.make_estimator()


@dataclass
class KNNRegressorBuilder(ModelBuilder):
    cfg: KNNRegressorConfig

    def make_estimator(self) -> SkRegressor:
        kw = _filtered_kwargs(KNeighborsRegressor, self.cfg)
        return KNeighborsRegressor(**kw)

    def build(self) -> SkRegressor:
        return self.make_estimator()


@dataclass
class DecisionTreeRegressorBuilder(ModelBuilder):
    cfg: DecisionTreeRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> SkRegressor:
        kw = _filtered_kwargs(DecisionTreeRegressor, self.cfg)
        _maybe_set_random_state(DecisionTreeRegressor, kw, self.seed)
        return DecisionTreeRegressor(**kw)

    def build(self) -> SkRegressor:
        return self.make_estimator()


@dataclass
class RandomForestRegressorBuilder(ModelBuilder):
    cfg: RandomForestRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> SkRegressor:
        kw = _filtered_kwargs(RandomForestRegressor, self.cfg)
        _maybe_set_random_state(RandomForestRegressor, kw, self.seed)
        return RandomForestRegressor(**kw)

    def build(self) -> SkRegressor:
        return self.make_estimator()

