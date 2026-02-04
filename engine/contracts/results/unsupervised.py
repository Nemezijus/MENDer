from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import ConfigDict, Field

from .common import ResultModel, JSONDict


class UnsupervisedOutputRow(ResultModel):
    """One per-sample unsupervised output row.

    Most models at minimum emit (index, cluster_id). Some may add extra
    per-sample fields; allow them.
    """

    model_config = ConfigDict(extra="allow")

    index: int
    cluster_id: int


class UnsupervisedOutputs(ResultModel):
    """Compact unsupervised per-sample outputs payload for Results UI."""

    notes: List[str] = Field(default_factory=list)
    preview_rows: List[UnsupervisedOutputRow] = Field(default_factory=list)
    n_rows_total: Optional[int] = None
    summary: Optional[JSONDict] = None


class UnsupervisedResult(ResultModel):
    task: Literal["unsupervised"] = "unsupervised"

    n_train: int
    n_features: int

    n_apply: Optional[int] = None

    metrics: Dict[str, Optional[float]] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

    cluster_summary: JSONDict = Field(default_factory=dict)
    diagnostics: JSONDict = Field(default_factory=dict)

    artifact: Optional[JSONDict] = None

    unsupervised_outputs: Optional[UnsupervisedOutputs] = None

    notes: List[str] = Field(default_factory=list)
