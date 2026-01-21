from typing import Optional, List, Union, Dict, Any, Literal

from pydantic import BaseModel, Field

from shared_schemas.types import MetricName
from shared_schemas.ensemble_configs import EnsembleConfig
from shared_schemas.run_config import DataModel
from shared_schemas.split_configs import SplitCVModel, SplitHoldoutModel
from shared_schemas.scale_configs import ScaleModel
from shared_schemas.feature_configs import FeaturesModel
from shared_schemas.eval_configs import EvalModel

from .model_artifact import ModelArtifactMeta
from .metrics_models import ConfusionMatrix, RocMetrics
from .decoder_models import DecoderOutputs




# -----------------------------------------------------------------------------
# Ensemble-specific reports (typed)
# -----------------------------------------------------------------------------

class VotingEstimatorSummary(BaseModel):
    name: str
    algo: str
    fold_scores: Optional[List[float]] = None
    mean: float
    std: float
    n: int


class VotingBestEstimatorSummary(BaseModel):
    name: Optional[str] = None
    mean: Optional[float] = None


class VotingAgreementReport(BaseModel):
    all_agree_rate: float
    pairwise_mean_agreement: Optional[float] = None
    labels: List[str]
    matrix: Optional[List[List[float]]] = None


class VotingHist(BaseModel):
    edges: List[float]
    counts: List[float]


class VotingVoteReport(BaseModel):
    mean_margin: float
    mean_strength: float
    tie_rate: float
    margin_hist: VotingHist
    strength_hist: VotingHist


class VotingChangeVsBestReport(BaseModel):
    best_name: Optional[str] = None
    total: int
    corrected: int
    harmed: int
    net: int
    disagreed: int


class VotingEnsembleClassificationReport(BaseModel):
    kind: Literal["voting"] = "voting"
    task: Literal["classification"] = "classification"

    metric_name: str
    voting: str
    n_estimators: int
    weights: Optional[List[float]] = None

    estimators: List[VotingEstimatorSummary]
    best_estimator: VotingBestEstimatorSummary
    agreement: VotingAgreementReport
    vote: VotingVoteReport
    change_vs_best: VotingChangeVsBestReport


class VotingSimilarityReport(BaseModel):
    labels: List[str]
    pairwise_corr: Optional[List[List[float]]] = None
    pairwise_absdiff: Optional[List[List[float]]] = None
    pairwise_mean_corr: Optional[float] = None
    pairwise_mean_absdiff: Optional[float] = None
    prediction_spread_mean: Optional[float] = None


class VotingRegressionErrorBlock(BaseModel):
    rmse: Optional[float] = None
    mae: Optional[float] = None
    median_ae: Optional[float] = None


class VotingRegressionBestBaseBlock(VotingRegressionErrorBlock):
    name: Optional[str] = None


class VotingRegressionGainBlock(BaseModel):
    rmse_reduction: Optional[float] = None
    mae_reduction: Optional[float] = None
    median_ae_reduction: Optional[float] = None


class VotingRegressionErrorsReport(BaseModel):
    n_total: int
    ensemble: VotingRegressionErrorBlock
    best_base: VotingRegressionBestBaseBlock
    gain_vs_best: VotingRegressionGainBlock


class VotingEnsembleRegressionReport(BaseModel):
    kind: Literal["voting"] = "voting"
    task: Literal["regression"] = "regression"

    metric_name: str
    n_estimators: int
    weights: Optional[List[float]] = None

    estimators: List[VotingEstimatorSummary]
    best_estimator: VotingBestEstimatorSummary
    similarity: VotingSimilarityReport
    errors: VotingRegressionErrorsReport


EnsembleReport = Union[
    VotingEnsembleClassificationReport,
    VotingEnsembleRegressionReport,
    Dict[str, Any],
]


class EnsembleTrainRequest(BaseModel):
    data: DataModel
    split: Union[SplitHoldoutModel, SplitCVModel]
    scale: ScaleModel
    features: FeaturesModel
    ensemble: EnsembleConfig
    eval: EvalModel


class EnsembleTrainResponse(BaseModel):
    metric_name: MetricName
    metric_value: float
    confusion: ConfusionMatrix
    n_train: int
    n_test: int
    notes: List[str] = Field(default_factory=list)

    # Classification only (None for regression)
    roc: Optional[RocMetrics] = None

    # Shuffle-baseline fields (optional; parity with single-model train)
    shuffled_scores: Optional[List[float]] = None
    p_value: Optional[float] = None

    # K-fold summary fields (optional; parity with single-model train)
    fold_scores: Optional[List[float]] = None
    mean_score: Optional[float] = None
    std_score: Optional[float] = None
    n_splits: Optional[int] = None

    # Optional artifact metadata (same shape as single-model for now)
    artifact: Optional[ModelArtifactMeta] = None

    # Ensemble-specific insights (present for some ensemble kinds, e.g. voting)
    ensemble_report: Optional[EnsembleReport] = None

    # Optional: per-sample decoder outputs (classification only)
    decoder_outputs: Optional[DecoderOutputs] = None