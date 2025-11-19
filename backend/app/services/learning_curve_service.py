# backend/app/services/learning_curve_service.py
from typing import Dict, Any
import math
import numpy as np
from sklearn.model_selection import StratifiedKFold, learning_curve

from shared_schemas.run_config import RunConfig
from utils.factories.data_loading_factory import make_data_loader
from utils.factories.sanity_factory import make_sanity_checker
from utils.factories.pipeline_factory import make_pipeline
from utils.permutations.rng import RngManager

from ..models.v1.learning_curve_models import LearningCurveRequest, LearningCurveResponse

def compute_learning_curve(req: LearningCurveRequest) -> LearningCurveResponse:
    # Build a RunConfig exactly like your CV router does (pydantic models work fine here)
    cfg = RunConfig(
        data=req.data,
        split=req.split,
        scale=req.scale,
        features=req.features,
        model=req.model,
        eval=req.eval,
    )

    # 1) Load & sanity-check (reuse strategies)
    loader = make_data_loader(cfg.data)
    X, y = loader.load()
    make_sanity_checker().check(X, y)

    # 2) RNG & CV (StratifiedKFold like your CV flow)
    rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))
    cv = StratifiedKFold(
        n_splits=cfg.split.n_splits,
        shuffle=cfg.split.shuffle,
        random_state=(rngm.child_seed("lc/split") if cfg.split.shuffle else None),
    )

    # 3) Estimator pipeline (scale -> feat -> clf)
    pipe = make_pipeline(cfg, rngm, stream="lc")

    # 4) Train sizes
    if req.train_sizes is not None:
        train_sizes = np.array(req.train_sizes)
    else:
        train_sizes = np.linspace(0.1, 1.0, req.n_steps)
    
    # 5) sklearn.learning_curve
    sizes_abs, train_scores, val_scores = learning_curve(
        estimator=pipe,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=cfg.eval.metric,
        n_jobs=req.n_jobs,
        shuffle=False,        # we already handle shuffling in the CV splitter
        return_times=False,
    )

    # 6) Aggregate
    train_mean = np.mean(train_scores, axis=1).tolist()
    train_std  = np.std(train_scores, axis=1).tolist()
    val_mean   = np.mean(val_scores, axis=1).tolist()
    val_std    = np.std(val_scores, axis=1).tolist()

    def _sanitize_floats(lst):
        out = []
        for v in lst:
            try:
                fv = float(v)
                out.append(fv if math.isfinite(fv) else None)
            except Exception:
                out.append(None)
        return out

    return LearningCurveResponse(
        train_sizes=sizes_abs.tolist(),
        train_scores_mean=_sanitize_floats(train_mean),
        train_scores_std=_sanitize_floats(train_std),
        val_scores_mean=_sanitize_floats(val_mean),
        val_scores_std=_sanitize_floats(val_std),
    )
