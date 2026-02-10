from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field

from .types import CalibrationMethod


PositiveClassLabel = Union[int, str]


class DecoderOutputsConfig(BaseModel):
    """Configuration for per-sample decoder outputs in *classification* tasks.

    These outputs are intended for neural decoding / trial-level analysis, where you
    often need the decision value (decoder score) and/or class probabilities on a
    second dataset, not only aggregate accuracy.
    """

    enabled: bool = False

    # What to compute / expose
    include_decision_scores: bool = True
    include_probabilities: bool = True
    include_margin: bool = True

    # Binary classification: allows the UI/user to define what "positive" means
    # (e.g. go=1), so P(go) is consistent even if classes_ ordering changes.
    positive_class_label: Optional[PositiveClassLabel] = None

    # Optional calibration for estimators without predict_proba.
    # NOTE: proper calibration should be fit on training data only.
    calibrate_probabilities: bool = False
    calibration_method: CalibrationMethod = "sigmoid"
    calibration_cv: int = Field(default=5, ge=2)

    # When True, backend/frontend may export a per-trial table (CSV) in addition
    # to the normal plots. Business logic can use this flag to decide whether
    # to assemble the full table.
    enable_export: bool = True
