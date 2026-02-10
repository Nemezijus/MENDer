"""Shared schema contracts.

This package contains Pydantic models and Literal-based choice types used to
validate configuration payloads across MENDer.

Export policy:
- Keep module imports explicit in most of the codebase:
    from engine.contracts.run_config import RunConfig
- The names re-exported here are intended as a small set of convenience
  imports for callers that prefer a single namespace.
"""

from ...engine.contracts.choices import (
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

from ...engine.contracts.decoder_configs import DecoderOutputsConfig
from ...engine.contracts.ensemble_configs import (
    AdaBoostEnsembleConfig,
    BaggingEnsembleConfig,
    EnsembleConfig,
    VotingEnsembleConfig,
    VotingEstimatorSpec,
    XGBoostEnsembleConfig,
)
from ...engine.contracts.ensemble_run_config import EnsembleRunConfig
from ...engine.contracts.eval_configs import EvalModel
from ...engine.contracts.feature_configs import FeaturesModel
from ...engine.contracts.metrics_configs import MetricsModel
from ...engine.contracts.model_configs import ModelConfig
from ...engine.contracts.run_config import DataModel, RunConfig
from ...engine.contracts.scale_configs import ScaleModel
from ...engine.contracts.split_configs import SplitCVModel, SplitHoldoutModel
from ...engine.contracts.tuning_configs import (
    GridSearchConfig,
    LearningCurveConfig,
    RandomizedSearchConfig,
    TuningConfig,
    ValidationCurveConfig,
)
from ...engine.contracts.unsupervised_configs import UnsupervisedEvalModel, UnsupervisedRunConfig

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
