from typing import Optional, List
from pydantic import BaseModel

from .model_artifact import ModelArtifactMeta
from .data_models import DataInspectRequest, Label


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

class ApplyModelExportRequest(ApplyModelRequest):
    """
    Request body for exporting predictions as CSV.

    Same as ApplyModelRequest, plus an optional filename hint for the server.
    """
    filename: Optional[str] = None