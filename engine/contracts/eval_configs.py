from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .types import MetricName
from .decoder_configs import DecoderOutputsConfig


class EvalModel(BaseModel):
    """Evaluation settings for supervised training and most tuning flows.

    Notes
    -----
    - ``metric`` is optional so the Engine can choose the correct default based
      on task kind (classification vs regression). This avoids UI-side "safe"
      defaulting and makes /schema/defaults the single source of truth.
    """

    metric: Optional[MetricName] = None
    seed: Optional[int] = None
    n_shuffles: int = 0

    # Optional progress id used by the shuffle-baseline progress endpoint
    progress_id: Optional[str] = None

    # Decoder outputs (classification-only). When enabled, training/apply can
    # compute per-sample decision values and/or probabilities.
    decoder: DecoderOutputsConfig = Field(default_factory=DecoderOutputsConfig)
