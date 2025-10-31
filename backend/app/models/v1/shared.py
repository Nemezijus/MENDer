from typing import Optional, Union, List, Literal
from pydantic import BaseModel

FeatureName = Literal["none", "pca", "lda", "sfs"]

class FeaturesModel(BaseModel):
    method: FeatureName = "none"

    # PCA
    pca_n: Optional[int] = None
    pca_var: float = 0.95
    pca_whiten: bool = False

    # LDA
    lda_n: Optional[int] = None
    lda_solver: Literal["svd", "lsqr", "eigen"] = "svd"
    # sklearn supports float or "auto" (for shrinkage with 'lsqr'/'eigen')
    lda_shrinkage: Optional[Union[float, Literal["auto"]]] = None
    lda_tol: float = 1e-4
    # Optional priors (list of class priors); default None
    lda_priors: Optional[List[float]] = None

    # SFS
    sfs_k: Union[int, Literal["auto"]] = "auto"
    sfs_direction: Literal["forward", "backward"] = "backward"
    sfs_cv: int = 5
    sfs_n_jobs: Optional[int] = None