from __future__ import annotations

import math
from typing import List

from sklearn.model_selection import KFold, StratifiedKFold

from engine.contracts.run_config import RunConfig
from engine.core.random.rng import RngManager


UNSUPERVISED_METRICS = {"silhouette", "davies_bouldin", "calinski_harabasz"}


def resolve_param_name_for_pipeline(pipe, raw_name: str) -> str:
    """Map a logical model parameter name (e.g. 'C') to a Pipeline parameter name.

    - If raw_name already exists in pipe.get_params(), it is returned unchanged.
    - Otherwise we try <last_step_name>__<raw_name>.
    - If that also doesn't exist, we return raw_name and let sklearn raise.
    """
    if not raw_name:
        return raw_name

    params = pipe.get_params(deep=True)
    if raw_name in params:
        return raw_name

    if getattr(pipe, "steps", None):
        last_step_name = pipe.steps[-1][0]
        candidate = f"{last_step_name}__{raw_name}"
        if candidate in params:
            return candidate

    return raw_name


def sanitize_floats(values: List[float]) -> List[float | None]:
    out: List[float | None] = []
    for v in values:
        try:
            fv = float(v)
            out.append(fv if math.isfinite(fv) else None)
        except Exception:
            out.append(None)
    return out


def coerce_unsupervised_metric(metric: str) -> str:
    return metric if metric in UNSUPERVISED_METRICS else "silhouette"


def cv_for_cfg(
    cfg: RunConfig,
    *,
    rngm: RngManager,
    stream: str,
    force_unstratified: bool = False,
):
    cv_seed = rngm.child_seed(f"{stream}/split") if cfg.split.shuffle else None

    if force_unstratified:
        return KFold(
            n_splits=cfg.split.n_splits,
            shuffle=cfg.split.shuffle,
            random_state=cv_seed,
        )

    stratified_flag = getattr(cfg.split, "stratified", True)
    if stratified_flag:
        return StratifiedKFold(
            n_splits=cfg.split.n_splits,
            shuffle=cfg.split.shuffle,
            random_state=cv_seed,
        )

    return KFold(
        n_splits=cfg.split.n_splits,
        shuffle=cfg.split.shuffle,
        random_state=cv_seed,
    )
