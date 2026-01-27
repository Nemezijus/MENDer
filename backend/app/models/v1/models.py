from typing import Optional, List
from pydantic import BaseModel

from .model_artifact import ModelArtifactMeta
from .data_models import DataInspectRequest, Label
from .decoder_models import DecoderOutputs
from shared_schemas.eval_configs import EvalModel
from shared_schemas.unsupervised_configs import UnsupervisedEvalModel


class SaveModelRequest(BaseModel):
    artifact_uid: str
    artifact_meta: ModelArtifactMeta
    filename: Optional[str] = None  # optional client-suggested name (e.g., "my_model.mend")


class LoadModelResponse(BaseModel):
    artifact: ModelArtifactMeta

# ---------------------------------------------------------------------------
# Apply model to production data
# ---------------------------------------------------------------------------

class ApplyModelRequest(BaseModel):
    """
    Request to apply an existing model artifact to a new dataset.

    - artifact_uid: identifies the cached pipeline (from train/load).
    - artifact_meta: the same meta dict used for save/load; needed so the
      service can reconstruct ModelConfig/EvalModel and enforce compatibility.
    - data: describes where to load X (and optional y) from, reusing the same
      structure as data inspection.
    """
    artifact_uid: str
    artifact_meta: ModelArtifactMeta
    data: DataInspectRequest

    # Optional override for evaluation/decoder settings during apply/export.
    # If provided, this will be used instead of artifact_meta.eval.
    eval: Optional[EvalModel] = None


class ApplyUnsupervisedModelRequest(BaseModel):
    """Request to apply an unsupervised (clustering) artifact to a new dataset.

    Notes
    -----
    - Labels (y) in the provided dataset are ignored for unsupervised apply.
    - Only predict-capable unsupervised models can be applied to unseen datasets.
    """

    artifact_uid: str
    artifact_meta: ModelArtifactMeta
    data: DataInspectRequest

    # Optional override for unsupervised evaluation settings.
    # This is currently informational (apply returns assignments only), but kept
    # for parity with the supervised ApplyModelRequest.
    eval: Optional[UnsupervisedEvalModel] = None


class PredictionRow(BaseModel):
    """
    One row of the prediction preview table.

    - index:      row index (0-based)
    - y_pred:     predicted label / value
    - y_true:     optional ground truth (if labels were provided)
    - residual:   for regression, y_true - y_pred (if numeric)
    - abs_error:  for regression, |residual|
    - correct:    for classification, whether prediction matches y_true
    """
    index: int
    y_pred: Label
    y_true: Optional[Label] = None
    residual: Optional[float] = None
    abs_error: Optional[float] = None
    correct: Optional[bool] = None


class ApplyModelResponse(BaseModel):
    """
    Response from applying a model to a new dataset.

    - n_samples:    number of samples in X
    - n_features:   number of features in X
    - task:         inferred task from the model ("classification", "regression", ...)
    - has_labels:   whether labels were supplied for this dataset
    - metric_name:  evaluation metric used, if any
    - metric_value: metric value on this dataset, if labels were provided
    - preview:      compact preview table for the first N rows
    - notes:        human-readable notes / warnings
    """
    n_samples: int
    n_features: int
    task: str
    has_labels: bool
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    preview: List[PredictionRow]
    notes: List[str] = []

    # Optional: per-sample decoder outputs preview (classification only)
    decoder_outputs: Optional[DecoderOutputs] = None


class UnsupervisedPredictionRow(BaseModel):
    """One row of the unsupervised apply preview table."""

    index: int
    cluster_id: int


class ApplyUnsupervisedModelResponse(BaseModel):
    """Response from applying an unsupervised artifact to a new dataset."""

    n_samples: int
    n_features: int
    task: str
    preview: List[UnsupervisedPredictionRow]
    notes: List[str] = []

class ApplyModelExportRequest(ApplyModelRequest):
    """
    Request body for exporting predictions as CSV.

    Same as ApplyModelRequest, plus an optional filename hint for the server.
    """
    filename: Optional[str] = None


class ApplyUnsupervisedModelExportRequest(ApplyUnsupervisedModelRequest):
    """Request body for exporting unsupervised predictions as CSV."""

    filename: Optional[str] = None


class ExportDecoderOutputsRequest(BaseModel):
    """Request body for exporting cached evaluation (decoder) outputs as CSV."""

    artifact_uid: str
    filename: Optional[str] = None