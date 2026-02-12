from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

def _filtered_kwargs(estimator_cls: type, cfg_obj: Any, *, exclude: set[str] = {"algo"}) -> Dict[str, Any]:
    """Dump cfg to dict, drop None, remove 'algo', and keep only kwargs accepted by estimator."""
    # Use aliases so config fields can safely avoid name collisions with BaseModel methods
    # (e.g., `copy_` field with alias "copy").
    raw = cfg_obj.model_dump(exclude=exclude, exclude_none=True, by_alias=True)
    sig = inspect.signature(estimator_cls)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in raw.items() if k in allowed}


def _maybe_set_random_state(estimator_cls: type, kw: Dict[str, Any], seed: Optional[int]) -> None:
    if seed is None:
        return
    sig = inspect.signature(estimator_cls)
    if "random_state" in sig.parameters and "random_state" not in kw:
        kw["random_state"] = int(seed)


