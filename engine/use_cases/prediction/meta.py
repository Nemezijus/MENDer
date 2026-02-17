from __future__ import annotations

from typing import Any, Optional

from engine.contracts.eval_configs import EvalModel

from engine.use_cases.prediction.utils import meta_get


def resolve_eval_model(
    *,
    artifact_meta: Any,
    eval_override: Optional[EvalModel] = None,
) -> Optional[EvalModel]:
    """Resolve EvalModel from an override or from stored artifact meta."""

    if eval_override is not None:
        return eval_override

    eval_dict = meta_get(artifact_meta, "eval", {}) or {}

    if not eval_dict:
        return None

    # Pydantic v2
    try:
        return EvalModel.model_validate(eval_dict)
    except Exception:
        pass

    # Pydantic v1 compatibility
    try:
        return EvalModel.parse_obj(eval_dict)  # type: ignore[attr-defined]
    except Exception:
        return None
