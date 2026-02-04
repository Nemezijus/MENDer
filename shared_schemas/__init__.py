"""Shared schema contracts.

This package contains Pydantic models and Literal-based choice types used to
validate configuration payloads across MENDer.

Export policy:
- Keep module imports explicit in most of the codebase:
    from shared_schemas.run_config import RunConfig
- The names re-exported here are intended as a small set of convenience
  imports for callers that prefer a single namespace.
"""

from .choices import (
    EnsembleKind,
    FeatureName,
    FitScopeName,
    LDASolver,
    MaxFeaturesName,
    MetricName,
    PenaltyName,
    ProblemKind,
    ScaleName,
    SVMDecisionShape,
    SVMKernel,
    TreeCriterion,
    TreeSplitter,
    TuningKind,
)

from .decoder_configs import DecoderOutputsConfig
from .ensemble_configs import (
    AdaBoostEnsembleConfig,
    BaggingEnsembleConfig,
    EnsembleConfig,
    VotingEnsembleConfig,
    VotingEstimatorSpec,
    XGBoostEnsembleConfig,
)
from .ensemble_run_config import EnsembleRunConfig
from .eval_configs import EvalModel
from .feature_configs import FeaturesModel
from .metrics_configs import MetricsModel
from .model_configs import ModelConfig
from .run_config import DataModel, RunConfig
from .scale_configs import ScaleModel
from .split_configs import SplitCVModel, SplitHoldoutModel
from .tuning_configs import (
    GridSearchConfig,
    LearningCurveConfig,
    RandomizedSearchConfig,
    TuningConfig,
    ValidationCurveConfig,
)
from .unsupervised_configs import UnsupervisedEvalModel, UnsupervisedRunConfig

__all__ = [
    # choice types
    "EnsembleKind",
    "FeatureName",
    "FitScopeName",
    "LDASolver",
    "MaxFeaturesName",
    "MetricName",
    "PenaltyName",
    "ProblemKind",
    "ScaleName",
    "SVMDecisionShape",
    "SVMKernel",
    "TreeCriterion",
    "TreeSplitter",
    "TuningKind",
    # core configs
    "DataModel",
    "RunConfig",
    "ModelConfig",
    "FeaturesModel",
    "ScaleModel",
    "SplitHoldoutModel",
    "SplitCVModel",
    "EvalModel",
    "DecoderOutputsConfig",
    "MetricsModel",
    # tuning configs
    "TuningConfig",
    "LearningCurveConfig",
    "ValidationCurveConfig",
    "GridSearchConfig",
    "RandomizedSearchConfig",
    # ensembles
    "VotingEstimatorSpec",
    "VotingEnsembleConfig",
    "BaggingEnsembleConfig",
    "AdaBoostEnsembleConfig",
    "XGBoostEnsembleConfig",
    "EnsembleConfig",
    "EnsembleRunConfig",
    # unsupervised
    "UnsupervisedEvalModel",
    "UnsupervisedRunConfig",
]
