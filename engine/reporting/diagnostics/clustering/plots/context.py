from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .deps import np


@dataclass
class PlotContext:
    """Shared, pre-validated inputs for clustering plot builders.

    The orchestrator is responsible for:
    - validating/aligning X and labels
    - computing downsample indices
    - resolving the final estimator

    Plot modules should be best-effort and never raise.
    """

    model: Any
    est: Any
    est_name: str

    Xa: Any  # ndarray-like (2D)
    y: Any  # ndarray-like (1D)
    n: int

    idx: Any  # ndarray-like (1D indices)

    per_sample: Optional[Mapping[str, Any]] = None
    embedding: Optional[Mapping[str, Any]] = None

    seed: int = 0

    warnings: List[str] = field(default_factory=list)
    cache: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_numpy(self) -> bool:
        return np is not None
