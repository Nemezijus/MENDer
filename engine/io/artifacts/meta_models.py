from __future__ import annotations

"""Typed internal contracts for artifact metadata.

These models are *Business Layer* helpers: they improve internal typing and make
it harder for regressions (e.g., accidentally leaking fields into result
contracts).

We keep these intentionally lightweight (TypedDict over Pydantic) because the
metadata is ultimately serialized as a plain dict.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TypedDict


class ArtifactPipelineStep(TypedDict, total=False):
    name: str
    class_path: str
    params: Dict[str, Any]


class ArtifactSummary(TypedDict, total=False):
    metric_name: Optional[str]
    metric_value: Optional[float]
    mean_score: Optional[float]
    std_score: Optional[float]
    n_splits: Optional[int]
    notes: List[str]
    # Extra stats to attach to artifact metadata (NOT to result contracts)
    extra_stats: Dict[str, Any]


class ModelArtifactMetaDict(TypedDict, total=False):
    uid: str
    created_at: datetime
    mender_version: Optional[str]
    kind: Literal["classification", "regression", "unsupervised"]
    n_samples_train: Optional[int]
    n_samples_test: Optional[int]
    n_features_in: Optional[int]
    classes: Optional[List[Any]]
    split: Dict[str, Any]
    scale: Optional[Dict[str, Any]]
    features: Optional[Dict[str, Any]]
    model: Optional[Dict[str, Any]]
    eval: Optional[Dict[str, Any]]
    pipeline: List[ArtifactPipelineStep]
    metric_name: Optional[str]
    metric_value: Optional[float]
    mean_score: Optional[float]
    std_score: Optional[float]
    n_splits: Optional[int]
    notes: List[str]
    n_parameters: Optional[int]
    extra_stats: Dict[str, Any]
