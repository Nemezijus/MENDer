from typing import Optional, List, Dict, Union
from pydantic import BaseModel, Field

# Request allows either npz_path or x_path+y_path
class DataInspectRequest(BaseModel):
    npz_path: Optional[str] = None
    x_key: Optional[str] = "X"
    y_key: Optional[str] = "y"

    x_path: Optional[str] = None
    y_path: Optional[str] = None

class Missingness(BaseModel):
    total: int
    by_column: Optional[List[int]] = None  # optional: only for X

class Suggestions(BaseModel):
    recommend_pca: bool
    reason: Optional[str] = None

Label = Union[int, float, str]

class DataInspectResponse(BaseModel):
    n_samples: int
    n_features: int
    classes: List[Label]
    class_counts: Dict[str, int]
    missingness: Missingness
    suggestions: Suggestions
