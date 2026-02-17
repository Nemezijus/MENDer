from __future__ import annotations

"""Positive-class resolution helpers."""

from typing import Any, Optional

import numpy as np


def resolve_positive_class_index(
    *,
    classes: Optional[np.ndarray],
    positive_class_label: Any,
    positive_class_index: Optional[int],
) -> Optional[int]:
    """Resolve positive class index from label/classes if not already given."""

    if positive_class_index is not None:
        try:
            return int(positive_class_index)
        except Exception:
            return None

    if positive_class_label is None or classes is None:
        return None

    try:
        matches = np.where(np.asarray(classes) == positive_class_label)[0]
        if matches.size > 0:
            return int(matches[0])
    except Exception:
        return None

    return None
