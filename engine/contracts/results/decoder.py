from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from .common import Label, ResultModel, JSONDict


class DecoderOutputRow(ResultModel):
    """One per-sample decoder-output row for preview tables.

    We allow extra columns so downstream callers can attach fields such as:
    - fold_id
    - per-class proba columns
    - per-class decision scores
    """

    model_config = ConfigDict(extra="allow")

    index: int
    y_pred: Label
    y_true: Optional[Label] = None

    # Optional confidence proxy used in UIs.
    margin: Optional[float] = None


class DecoderSummary(ResultModel):
    """Compact global summaries computed from per-sample decoder outputs.

    This is intentionally flexible: different models/tasks can emit different
    summary fields, but we reserve a few common names.
    """

    model_config = ConfigDict(extra="allow")

    log_loss: Optional[float] = None
    brier: Optional[float] = None
    margin_mean: Optional[float] = None
    max_proba_mean: Optional[float] = None


class DecoderOutputs(ResultModel):
    """Compact decoder-outputs payload for Results UIs.

    This mirrors the backend API shape but lives in BL contracts.
    """

    classes: Optional[List[Label]] = None

    positive_class_label: Optional[Label] = None
    positive_class_index: Optional[int] = None

    has_decision_scores: bool = False
    has_proba: bool = False

    # How probabilities were obtained: 'model_proba' or 'vote_share'.
    proba_source: Optional[str] = None

    notes: List[str] = Field(default_factory=list)

    preview_rows: List[DecoderOutputRow] = Field(default_factory=list)
    n_rows_total: Optional[int] = None

    # Allow either a typed summary object or a raw dict (forward compat)
    summary: Optional[Dict[str, Any]] = None

    @classmethod
    def from_payload(cls, payload: JSONDict) -> "DecoderOutputs":
        return cls.model_validate(payload)
