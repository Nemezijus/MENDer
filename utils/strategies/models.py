# utils/strategies/models.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import inspect

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    RidgeCV,
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    BayesianRidge,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from sklearn.naive_bayes import GaussianNB

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
from utils.strategies.interfaces import ModelBuilder


def _filtered_kwargs(estimator_cls: type, cfg_obj: Any, *, exclude: set[str] = {"algo"}) -> Dict[str, Any]:
    """Dump cfg to dict, drop None, remove 'algo', and keep only kwargs accepted by estimator."""
    raw = cfg_obj.model_dump(exclude=exclude, exclude_none=True)
    sig = inspect.signature(estimator_cls)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in raw.items() if k in allowed}


def _maybe_set_random_state(estimator_cls: type, kw: Dict[str, Any], seed: Optional[int]) -> None:
    if seed is None:
        return
    sig = inspect.signature(estimator_cls)
    if "random_state" in sig.parameters and "random_state" not in kw:
        kw["random_state"] = int(seed)


@dataclass
class LogRegBuilder(ModelBuilder):
    cfg: LogRegConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        # Start from filtered kwargs
        kw = _filtered_kwargs(LogisticRegression, self.cfg)
        _maybe_set_random_state(LogisticRegression, kw, self.seed)
        # sklearn quirks: penalty/solver/l1_ratio interplay
        penalty = self.cfg.penalty
        solver = self.cfg.solver

        sk_penalty: Optional[str]
        if penalty == "none":
            sk_penalty = None
        else:
            sk_penalty = penalty

        if sk_penalty is None:
            if solver == "liblinear":
                solver = "lbfgs"
        elif sk_penalty == "elasticnet":
            solver = "saga"
        elif sk_penalty == "l1" and solver not in ("liblinear", "saga"):
            solver = "saga"

        # Overwrite with the normalized values
        kw.update({
            "penalty": sk_penalty,
            "solver": solver,
            "multi_class": "auto",
        })

        # l1_ratio only if elasticnet, else remove if present
        if sk_penalty == "elasticnet":
            kw["l1_ratio"] = self.cfg.l1_ratio
        else:
            kw.pop("l1_ratio", None)

        return LogisticRegression(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class SVMBuilder(ModelBuilder):
    cfg: SVMConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(SVC, self.cfg)
        _maybe_set_random_state(SVC, kw, self.seed)
        return SVC(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class DecisionTreeBuilder(ModelBuilder):
    cfg: TreeConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(DecisionTreeClassifier, self.cfg)
        _maybe_set_random_state(DecisionTreeClassifier, kw, self.seed)
        return DecisionTreeClassifier(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class RandomForestBuilder(ModelBuilder):
    cfg: ForestConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(RandomForestClassifier, self.cfg)
        _maybe_set_random_state(RandomForestClassifier, kw, self.seed)
        return RandomForestClassifier(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class KNNBuilder(ModelBuilder):
    cfg: KNNConfig

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(KNeighborsClassifier, self.cfg)
        return KNeighborsClassifier(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class GaussianNBBuilder(ModelBuilder):
    cfg: GaussianNBConfig

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(GaussianNB, self.cfg)
        return GaussianNB(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class RidgeClassifierBuilder(ModelBuilder):
    cfg: RidgeClassifierConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(RidgeClassifier, self.cfg)
        _maybe_set_random_state(RidgeClassifier, kw, self.seed)
        return RidgeClassifier(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class SGDClassifierBuilder(ModelBuilder):
    cfg: SGDClassifierConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(SGDClassifier, self.cfg)
        _maybe_set_random_state(SGDClassifier, kw, self.seed)
        return SGDClassifier(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class ExtraTreesBuilder(ModelBuilder):
    cfg: ExtraTreesConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(ExtraTreesClassifier, self.cfg)
        _maybe_set_random_state(ExtraTreesClassifier, kw, self.seed)
        return ExtraTreesClassifier(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class HistGradientBoostingBuilder(ModelBuilder):
    cfg: HistGradientBoostingConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(HistGradientBoostingClassifier, self.cfg)
        _maybe_set_random_state(HistGradientBoostingClassifier, kw, self.seed)
        return HistGradientBoostingClassifier(**kw)

    def build(self) -> Any:
        return self.make_estimator()


"""Regressor builders.

These mirror the classifier builders above: config -> filtered kwargs -> estimator instance.
"""


# -------------REGRESSORS----------------


@dataclass
class LinRegBuilder(ModelBuilder):
    cfg: LinearRegConfig

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(LinearRegression, self.cfg)
        return LinearRegression(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class RidgeRegressorBuilder(ModelBuilder):
    cfg: RidgeRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(Ridge, self.cfg)
        _maybe_set_random_state(Ridge, kw, self.seed)
        return Ridge(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class RidgeCVRegressorBuilder(ModelBuilder):
    cfg: RidgeCVRegressorConfig

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(RidgeCV, self.cfg)
        return RidgeCV(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class ElasticNetRegressorBuilder(ModelBuilder):
    cfg: ElasticNetRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(ElasticNet, self.cfg)
        _maybe_set_random_state(ElasticNet, kw, self.seed)
        return ElasticNet(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class ElasticNetCVRegressorBuilder(ModelBuilder):
    cfg: ElasticNetCVRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(ElasticNetCV, self.cfg)
        _maybe_set_random_state(ElasticNetCV, kw, self.seed)
        return ElasticNetCV(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class LassoRegressorBuilder(ModelBuilder):
    cfg: LassoRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(Lasso, self.cfg)
        _maybe_set_random_state(Lasso, kw, self.seed)
        return Lasso(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class LassoCVRegressorBuilder(ModelBuilder):
    cfg: LassoCVRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(LassoCV, self.cfg)
        _maybe_set_random_state(LassoCV, kw, self.seed)
        return LassoCV(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class BayesianRidgeRegressorBuilder(ModelBuilder):
    cfg: BayesianRidgeRegressorConfig

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(BayesianRidge, self.cfg)
        return BayesianRidge(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class SVRRegressorBuilder(ModelBuilder):
    cfg: SVRRegressorConfig

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(SVR, self.cfg)
        return SVR(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class LinearSVRRegressorBuilder(ModelBuilder):
    cfg: LinearSVRRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(LinearSVR, self.cfg)
        _maybe_set_random_state(LinearSVR, kw, self.seed)
        return LinearSVR(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class KNNRegressorBuilder(ModelBuilder):
    cfg: KNNRegressorConfig

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(KNeighborsRegressor, self.cfg)
        return KNeighborsRegressor(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class DecisionTreeRegressorBuilder(ModelBuilder):
    cfg: DecisionTreeRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(DecisionTreeRegressor, self.cfg)
        _maybe_set_random_state(DecisionTreeRegressor, kw, self.seed)
        return DecisionTreeRegressor(**kw)

    def build(self) -> Any:
        return self.make_estimator()


@dataclass
class RandomForestRegressorBuilder(ModelBuilder):
    cfg: RandomForestRegressorConfig
    seed: Optional[int] = None

    def make_estimator(self) -> Any:
        kw = _filtered_kwargs(RandomForestRegressor, self.cfg)
        _maybe_set_random_state(RandomForestRegressor, kw, self.seed)
        return RandomForestRegressor(**kw)

    def build(self) -> Any:
        return self.make_estimator()
