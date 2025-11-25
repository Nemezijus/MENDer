from __future__ import annotations

from typing import Optional, Union, Literal, List
from pydantic import BaseModel, field_validator

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
    lda_priors: Optional[List[float]] = None

    # SFS (Sequential Feature Selection)
    sfs_k: Union[int, Literal["auto"]] = "auto"
    sfs_direction: Literal["forward", "backward"] = "forward"
    sfs_cv: int = 5
    sfs_n_jobs: Optional[int] = None

    @field_validator("pca_n", "lda_n", "sfs_n_jobs", mode="before")
    @classmethod
    def _empty_to_none_or_int(cls, v):
        """
        Allow UI to send "", null, or a number.
        - "" / None -> None
        - anything else -> int(value)
        """
        if v is None or v == "":
            return None
        return int(v)

    @field_validator("sfs_cv", mode="before")
    @classmethod
    def _sfs_cv_default(cls, v):
        """
        Allow "", null, or int for sfs_cv.
        - "" / None -> 5 (default)
        - else -> int(value)
        """
        if v is None or v == "":
            return 5
        return int(v)

    @field_validator("sfs_k", mode="before")
    @classmethod
    def _sfs_k_auto_or_int(cls, v):
        """
        Accept typical UI values for sfs_k:
        - "" / None -> "auto"
        - "auto"    -> "auto"
        - numeric string / number -> int
        """
        if v is None or v == "":
            return "auto"

        if isinstance(v, str):
            v = v.strip()
            if v.lower() == "auto":
                return "auto"
            # numeric string -> int
            return int(v)

        # already a number
        return int(v)