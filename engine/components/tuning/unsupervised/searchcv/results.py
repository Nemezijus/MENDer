from __future__ import annotations

"""Internal result typing for unsupervised SearchCV."""

from typing import Any, Dict, List, TypedDict


class CVResults(TypedDict, total=False):
    """A minimal sklearn-like cv_results_ payload (frontend-facing)."""

    mean_score: List[float | None]
    std_score: List[float | None]
    mean_test_score: List[float | None]
    std_test_score: List[float | None]

    # Any number of param_* entries
    # param_<name>: List[Any]


CVResultsDict = Dict[str, Any]
