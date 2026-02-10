from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

from .types import MetricName
from .decoder_configs import DecoderOutputsConfig

class EvalModel(BaseModel):
    metric: MetricName = "accuracy"
    seed: Optional[int] = None
    n_shuffles: int = 0
    # Optional progress id used by the shuffle-baseline progress endpoint
    progress_id: Optional[str] = None
    # Decoder outputs (classification-only). When enabled, training/apply can
    # compute per-sample decision values and/or probabilities.
    decoder: DecoderOutputsConfig = Field(default_factory=DecoderOutputsConfig)
