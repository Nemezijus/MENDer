from __future__ import annotations

from typing import Optional
from pydantic import BaseModel

from .types import MetricName

class EvalModel(BaseModel):
    metric: MetricName = "accuracy"
    seed: Optional[int] = None
    n_shuffles: int = 0
    # Optional progress id used by the shuffle-baseline progress endpoint
    progress_id: Optional[str] = None
