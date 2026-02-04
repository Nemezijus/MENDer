from __future__ import annotations

"""Result contracts for the Business Layer.

These models represent *outputs* produced by business logic and/or orchestration
(use-cases) and are intended to be stable across backend/frontend changes.

Design goals:
- JSON-friendly field types (lists, dicts, scalars) at the contract boundary.
- Strict top-level validation (extra fields forbidden) to prevent silent drift.
- Forward compatibility where needed (row/record models allow extra columns).

Note: contracts should only depend on stdlib + pydantic.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

# Labels are allowed to be numbers or strings.
Label = Union[int, float, str]


class ResultModel(BaseModel):
    """Base class for BL result contracts (strict by default)."""

    model_config = ConfigDict(extra="forbid")


JSONDict = Dict[str, Any]
JSONList = List[Any]


def as_json_dict(v: Any) -> JSONDict:
    """Best-effort cast to a JSON-friendly dict."""
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    # Pydantic models
    if hasattr(v, "model_dump"):
        try:
            return v.model_dump()
        except Exception:
            pass
    # Dataclasses / objects
    if hasattr(v, "__dict__"):
        try:
            return dict(v.__dict__)
        except Exception:
            pass
    return {"value": v}
