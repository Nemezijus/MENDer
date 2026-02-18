from __future__ import annotations

from typing import Any, Dict

from engine.reporting.common.report_errors import record_error

from .context import PlotContext
from .deps import np


def add_gmm_ellipses(out: Dict[str, Any], ctx: PlotContext) -> None:
    """Optional covariance ellipses in embedding space.

    The frontend will render these as "rings" around clusters/components.
    Kept compatible with the pre-refactor payload shape.
    """
    if np is None:
        return

    try:
        if ctx.embedding is None:
            return

        ex = ctx.embedding.get("x", None)
        ey = ctx.embedding.get("y", None)
        eidx = ctx.embedding.get("idx", None)

        if not (
            isinstance(ex, list)
            and isinstance(ey, list)
            and isinstance(eidx, list)
            and len(ex) == len(ey) == len(eidx)
        ):
            return

        eidx_a = np.asarray(eidx, dtype=int)
        if not eidx_a.size or int(np.max(eidx_a)) >= ctx.n:
            return

        labs = ctx.y[eidx_a]
        comps = []
        ex_a = np.asarray(ex, dtype=float)
        ey_a = np.asarray(ey, dtype=float)

        for cid in sorted(set(int(v) for v in labs.tolist() if int(v) != -1)):
            msk = labs == cid
            if int(np.sum(msk)) < 5:
                continue

            pts = np.column_stack([ex_a[msk], ey_a[msk]])
            mu = np.mean(pts, axis=0)
            cov = np.cov(pts.T)

            if cov.shape == (2, 2) and np.all(np.isfinite(cov)):
                comps.append(
                    {
                        "cluster_id": int(cid),
                        "mean": [float(mu[0]), float(mu[1])],
                        "cov": [
                            [float(cov[0, 0]), float(cov[0, 1])],
                            [float(cov[1, 0]), float(cov[1, 1])],
                        ],
                    }
                )

        # Keep the pre-refactor behavior: only emit when the estimator
        # supports probabilistic membership (mixture-like models).
        if comps and hasattr(ctx.est, "predict_proba"):
            out["gmm_ellipses"] = {"components": comps}

    except Exception as e:
        record_error(out, where="reporting.clustering.plots.gmm_ellipses", exc=e)
        return
