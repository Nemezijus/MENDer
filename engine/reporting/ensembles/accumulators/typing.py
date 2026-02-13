from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class EstimatorScoreSummary(TypedDict, total=False):
    name: str
    algo: str
    fold_scores: Optional[List[float]]
    mean: Optional[float]
    std: Optional[float]
    n: int


class AgreementSection(TypedDict, total=False):
    all_agree_rate: Optional[float]
    pairwise_mean_agreement: Optional[float]
    labels: List[str]
    matrix: Optional[List[List[float]]]
