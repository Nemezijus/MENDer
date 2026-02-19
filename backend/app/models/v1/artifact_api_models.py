"""API models for model-artifact operations.

This module groups request/response schemas for endpoints that operate on
persisted model artifacts:

- save / load
- apply to new data
- export predictions / decoder outputs

Kept separate from ``model_artifact.py`` (which describes the artifact itself)
to make the API surface easier to navigate.
"""

from typing import Optional, List

from pydantic import BaseModel, Field

from .model_artifact import ModelArtifactMeta
from .data_models import DataInspectRequest, Label
from .decoder_models import DecoderOutputs
from engine.contracts.eval_configs import EvalModel
from engine.contracts.unsupervised_configs import UnsupervisedEvalModel


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
    """Request to apply an existing model artifact to a new dataset.

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
    """One row of the prediction preview table."""

    index: int
    y_pred: Label
    y_true: Optional[Label] = None
    residual: Optional[float] = None
    abs_error: Optional[float] = None
    correct: Optional[bool] = None


class ApplyModelResponse(BaseModel):
    """Response from applying a model to a new dataset."""

    n_samples: int
    n_features: int
    task: str
    has_labels: bool
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    preview: List[PredictionRow]
    notes: List[str] = Field(default_factory=list)

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
    notes: List[str] = Field(default_factory=list)


class ApplyModelExportRequest(ApplyModelRequest):
    """Request body for exporting predictions as CSV."""

    filename: Optional[str] = None


class ApplyUnsupervisedModelExportRequest(ApplyUnsupervisedModelRequest):
    """Request body for exporting unsupervised predictions as CSV."""

    filename: Optional[str] = None


class ExportDecoderOutputsRequest(BaseModel):
    """Request body for exporting cached evaluation (decoder) outputs as CSV."""

    artifact_uid: str
    filename: Optional[str] = None
