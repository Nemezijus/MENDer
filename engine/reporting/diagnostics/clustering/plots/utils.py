from __future__ import annotations

from typing import Any, Mapping, Optional

from engine.reporting.common.hist import histogram_centers_payload

from .deps import np


def histogram_payload(values: Any, *, bins: int = 30) -> Optional[Mapping[str, Any]]:
    """Return a small histogram payload {x, y} for JSON transport."""

    # Keep behavior identical to the historical implementation by delegating to the
    # shared primitive while still respecting optional dependency loading.
    return histogram_centers_payload(values, bins=bins, np_mod=np)
