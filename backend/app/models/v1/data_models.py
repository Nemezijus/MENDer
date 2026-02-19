from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# Request allows either npz_path or x_path (+ optional y_path)
class DataInspectRequest(BaseModel):
    # NPZ (single file) input
    npz_path: Optional[str] = None
    x_key: str = "X"
    y_key: str = "y"

    # Separate files input
    x_path: Optional[str] = None
    y_path: Optional[str] = None

    # Optional: allow backend adapter to transpose X if needed
    expected_n_features: Optional[int] = None


class Missingness(BaseModel):
    total: int
    by_column: List[int] = Field(default_factory=list)


class Suggestions(BaseModel):
    recommend_pca: bool
    reason: Optional[str] = None


Label = Union[int, float, str]


class DataInspectResponse(BaseModel):
    n_samples: int
    n_features: int

    # Legacy fields (kept for backwards compatibility with older frontend code)
    classes: List[Label] = Field(default_factory=list)
    class_counts: Dict[str, int] = Field(default_factory=dict)

    missingness: Missingness
    suggestions: Suggestions

    # Additional engine-derived fields
    task_inferred: Optional[str] = None
    y_summary: Dict[str, Any] = Field(default_factory=dict)
