"""Model estimator builders.

This package contains SRP-friendly builder classes that convert typed config
objects into concrete sklearn estimators.

Public API remains stable:
    from engine.components.models.builders import LogRegBuilder, ...
"""

from .classification import (
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
)

from .regression import (
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

from .unsupervised import (
    KMeansBuilder,
    DBSCANBuilder,
    SpectralClusteringBuilder,
    AgglomerativeClusteringBuilder,
    GaussianMixtureBuilder,
    BayesianGaussianMixtureBuilder,
    MeanShiftBuilder,
    BirchBuilder,
)

__all__ = [
    "LogRegBuilder",
    "SVMBuilder",
    "DecisionTreeBuilder",
    "RandomForestBuilder",
    "KNNBuilder",
    "GaussianNBBuilder",
    "RidgeClassifierBuilder",
    "SGDClassifierBuilder",
    "ExtraTreesBuilder",
    "HistGradientBoostingBuilder",
    "LinRegBuilder",
    "RidgeRegressorBuilder",
    "RidgeCVRegressorBuilder",
    "ElasticNetRegressorBuilder",
    "ElasticNetCVRegressorBuilder",
    "LassoRegressorBuilder",
    "LassoCVRegressorBuilder",
    "BayesianRidgeRegressorBuilder",
    "SVRRegressorBuilder",
    "LinearSVRRegressorBuilder",
    "KNNRegressorBuilder",
    "DecisionTreeRegressorBuilder",
    "RandomForestRegressorBuilder",
    "KMeansBuilder",
    "DBSCANBuilder",
    "SpectralClusteringBuilder",
    "AgglomerativeClusteringBuilder",
    "GaussianMixtureBuilder",
    "BayesianGaussianMixtureBuilder",
    "MeanShiftBuilder",
    "BirchBuilder",
]
