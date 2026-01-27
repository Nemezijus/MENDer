from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

from .types import UnsupervisedMetricName
from .run_config import DataModel
from .scale_configs import ScaleModel
from .feature_configs import FeaturesModel
from .model_configs import ModelConfig

# Default post-fit metric pack.
# Notes:
# - Each metric is computed *after* fitting, from X and produced labels.
# - Some metrics are undefined for degenerate solutions (e.g., <2 clusters).
DEFAULT_CLUSTERING_METRICS: List[UnsupervisedMetricName] = [
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
]


FitScopeName = Literal["train_only", "train_and_predict"]


class UnsupervisedEvalModel(BaseModel):
    """Evaluation / diagnostics settings for clustering.

    If `metrics` is empty, the service should compute DEFAULT_CLUSTERING_METRICS
    when they are well-defined.
    """

    metrics: List[UnsupervisedMetricName] = Field(default_factory=list)
    seed: Optional[int] = None

    # Convenience outputs for the UI
    compute_embedding_2d: bool = True
    embedding_method: Literal["pca"] = "pca"  # keep sklearn-only in v1

    # Per-sample outputs are useful for export and diagnostics.
    per_sample_outputs: bool = True

    # Optional wide outputs. Keep False by default to avoid huge payloads.
    include_cluster_probabilities: bool = False


class UnsupervisedRunConfig(BaseModel):
    """End-to-end config for unsupervised (clustering) training.

    - `data` points to the training dataset (X is required; y may exist but is ignored).
    - `apply` is optional "production" / unseen X to assign to existing clusters.
      Only some algorithms support `predict`; callers should gate this in the UI.
    """

    task: Literal["unsupervised"] = "unsupervised"

    data: DataModel
    apply: Optional[DataModel] = None

    fit_scope: FitScopeName = "train_only"

    scale: ScaleModel
    features: FeaturesModel
    model: ModelConfig
    eval: UnsupervisedEvalModel = Field(default_factory=UnsupervisedEvalModel)

    # Future: allow y-based "external" validation (ARI/NMI/etc.) when y is provided.
    use_y_for_external_metrics: bool = False
    external_metrics: List[str] = Field(default_factory=list)
