from __future__ import annotations

from typing import Optional, Union, Literal
from pydantic import BaseModel

from .types import FeatureName, LDASolver

class FeaturesModel(BaseModel):
    method: FeatureName = "none"

    # PCA
    pca_n: Optional[int] = None
    pca_var: float = 0.95
    pca_whiten: bool = False

    # LDA
    lda_n: Optional[int] = None
    lda_solver: LDASolver = "svd"
    lda_shrinkage: Optional[float] = None  # only for lsqr/eigen
    lda_tol: float = 1e-4

    # SFS (Sequential Feature Selection)
    sfs_k: Union[int, Literal["auto"]] = "auto"
    sfs_direction: Literal["forward", "backward"] = "forward"
    sfs_cv: int = 5
    sfs_n_jobs: Optional[int] = None
