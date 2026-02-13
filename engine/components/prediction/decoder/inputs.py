from __future__ import annotations

"""Input normalization helpers for decoder outputs."""

from typing import Any, Iterable, List, Optional, Sequence

import numpy as np


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for s in items:
        ss = str(s)
        if ss not in seen:
            seen.add(ss)
            out.append(ss)
    return out


def ensure_indices(indices: Optional[Iterable[int]], n: int) -> List[int]:
    if indices is None:
        return list(range(int(n)))
    try:
        idx = [int(i) for i in list(indices)]
    except Exception:
        return list(range(int(n)))
    if len(idx) != int(n):
        return list(range(int(n)))
    return idx


def normalize_classes(classes: Any | None) -> Optional[List[Any]]:
    if classes is None:
        return None
    try:
        arr = np.asarray(classes)
        return [c.item() if isinstance(c, np.generic) else c for c in arr.tolist()]
    except Exception:
        try:
            return list(classes)  # type: ignore[arg-type]
        except Exception:
            return None
