from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from .linear import (
    BayesianRidgeRegressorConfig,
    ElasticNetCVRegressorConfig,
    ElasticNetRegressorConfig,
    LassoCVRegressorConfig,
    LassoRegressorConfig,
    LinearRegConfig,
    LinearSVRRegressorConfig,
    LogRegConfig,
    RidgeCVRegressorConfig,
    RidgeClassifierConfig,
    RidgeRegressorConfig,
    SGDClassifierConfig,
)
from .svm import SVMConfig, SVRRegressorConfig
from .trees import (
    DecisionTreeRegressorConfig,
    ExtraTreesConfig,
    ForestConfig,
    HistGradientBoostingConfig,
    RandomForestRegressorConfig,
    TreeConfig,
)
from .neighbors import KNNConfig, KNNRegressorConfig
from .naive_bayes import GaussianNBConfig
from .unsupervised import (
    AgglomerativeClusteringConfig,
    BayesianGaussianMixtureConfig,
    BirchConfig,
    DBSCANConfig,
    GaussianMixtureConfig,
    KMeansConfig,
    MeanShiftConfig,
    SpectralClusteringConfig,
)


ModelConfig = Annotated[
    Union[
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
        KMeansConfig,
        DBSCANConfig,
        SpectralClusteringConfig,
        AgglomerativeClusteringConfig,
        GaussianMixtureConfig,
        BayesianGaussianMixtureConfig,
        MeanShiftConfig,
        BirchConfig,
    ],
    Field(discriminator="algo"),
]


def get_model_task_by_algo(algo: str) -> str:
    mapping = {
        "logreg": LogRegConfig.task,
        "svm": SVMConfig.task,
        "tree": TreeConfig.task,
        "forest": ForestConfig.task,
        "knn": KNNConfig.task,
        "gnb": GaussianNBConfig.task,
        "ridge": RidgeClassifierConfig.task,
        "sgd": SGDClassifierConfig.task,
        "extratrees": ExtraTreesConfig.task,
        "hgb": HistGradientBoostingConfig.task,
        "linreg": LinearRegConfig.task,
        "ridgereg": RidgeRegressorConfig.task,
        "ridgecv": RidgeCVRegressorConfig.task,
        "enet": ElasticNetRegressorConfig.task,
        "enetcv": ElasticNetCVRegressorConfig.task,
        "lasso": LassoRegressorConfig.task,
        "lassocv": LassoCVRegressorConfig.task,
        "bayridge": BayesianRidgeRegressorConfig.task,
        "svr": SVRRegressorConfig.task,
        "linsvr": LinearSVRRegressorConfig.task,
        "knnreg": KNNRegressorConfig.task,
        "treereg": DecisionTreeRegressorConfig.task,
        "rfreg": RandomForestRegressorConfig.task,
        "kmeans": KMeansConfig.task,
        "dbscan": DBSCANConfig.task,
        "spectral": SpectralClusteringConfig.task,
        "agglo": AgglomerativeClusteringConfig.task,
        "gmm": GaussianMixtureConfig.task,
        "bgmm": BayesianGaussianMixtureConfig.task,
        "meanshift": MeanShiftConfig.task,
        "birch": BirchConfig.task,
    }
    return mapping.get(algo, "classification")
