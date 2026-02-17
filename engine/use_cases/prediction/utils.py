from __future__ import annotations

import math
from typing import Any, Optional


def meta_get(meta: Any, key: str, default: Any = None) -> Any:
    """Read a field from meta which may be a dataclass-like object or dict."""

    if meta is None:
        return default
    if hasattr(meta, key):
        return getattr(meta, key)
    if isinstance(meta, dict):
        return meta.get(key, default)
    return default


def safe_float_optional(v: Any) -> Optional[float]:
    """Convert to finite float or return None."""

    try:
        f = float(v)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f
