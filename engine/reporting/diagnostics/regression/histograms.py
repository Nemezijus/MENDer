from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

from engine.reporting.common.hist import histogram_edges_counts_payload

from .common import numpy


def histogram_1d(
    values: Any,
    *,
    n_bins: int = 30,
    value_range: Optional[Tuple[float, float]] = None,
) -> Optional[Mapping[str, List[float]]]:
    """Compute a compact histogram {edges, counts}.

    Returns None if it cannot be computed.
    """
    np = numpy()
    return histogram_edges_counts_payload(values, n_bins=n_bins, value_range=value_range, np_mod=np)