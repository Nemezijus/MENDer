from __future__ import annotations

from typing import Any, Optional

import numpy as np

from engine.reporting.ensembles.common import vote_margin_and_strength


def oob_coverage_from_decision_function(oob_decision: Any) -> Optional[float]:
    """Estimate OOB coverage from sklearn-style oob_decision_function_.

    Typically shape: (n_train, n_classes) with NaNs for samples that never had OOB preds.
    Coverage = fraction of rows that have no NaNs.
    """
    try:
        a = np.asarray(oob_decision)
        if a.ndim < 2:
            return None
        ok = ~np.any(np.isnan(a), axis=1)
        if ok.size == 0:
            return None
        return float(np.mean(ok))
    except Exception:
        return None

__all__ = ["vote_margin_and_strength", "oob_coverage_from_decision_function"]
