from __future__ import annotations

from typing import Optional, List, Union, Dict, Any

from pydantic import BaseModel, Field, ConfigDict

from .data_models import Label


class DecoderOutputRow(BaseModel):
    """One per-sample decoder-output row (used for preview tables).

    Notes on shapes:
      - Binary classification:
          decision_scores: float
          proba:           List[float] with len == n_classes (usually 2)
      - Multiclass classification:
          decision_scores: List[float] with len == n_classes (if available)
          proba:           List[float] with len == n_classes (if available)

    The `classes` ordering is provided in the parent payload.
    """
    model_config = ConfigDict(extra="allow")

    index: int
    y_pred: Label
    y_true: Optional[Label] = None

    margin: Optional[float] = None


class DecoderOutputs(BaseModel):
    """Compact decoder-outputs payload for Results UI.

    To keep API responses reasonably small, services should populate `preview_rows`
    (first N samples) and optionally include `n_rows_total` for the full dataset.
    CSV export can be provided via a separate endpoint.
    """

    classes: Optional[List[Label]] = None

    positive_class_label: Optional[Label] = None
    positive_class_index: Optional[int] = None

    has_decision_scores: bool = False
    has_proba: bool = False


    # How probabilities were obtained: 'model_proba' or 'vote_share'
    proba_source: Optional[str] = None
    notes: List[str] = Field(default_factory=list)

    # Preview table rows (first N samples)
    preview_rows: List[DecoderOutputRow] = Field(default_factory=list)
    n_rows_total: Optional[int] = None

    # Compact global summaries computed from per-sample decoder outputs.
    # Expected to be derived from OOF predictions when CV is used.
    summary: Optional[Dict[str, Any]] = None
