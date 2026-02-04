from __future__ import annotations

from typing import Any, Dict


def to_payload(obj: Any) -> Dict[str, Any]:
    """Coerce a result object into a JSON-friendly mapping.

    Supports:
      - Pydantic v2 models (model_dump)
      - Pydantic v1 models (dict)
      - Plain dicts

    This is intentionally small and backend-local: the BL owns the contracts;
    the backend just serializes them at the API boundary.
    """

    if obj is None:
        return {}

    # Pydantic v2
    md = getattr(obj, "model_dump", None)
    if callable(md):
        return md()

    # Pydantic v1
    d = getattr(obj, "dict", None)
    if callable(d):
        return d()

    if isinstance(obj, dict):
        return obj

    raise TypeError(f"Unsupported result payload type: {type(obj).__name__}")
