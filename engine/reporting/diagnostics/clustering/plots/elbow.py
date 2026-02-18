from __future__ import annotations

from typing import Any, Dict

from engine.reporting.common.report_errors import record_error

from .context import PlotContext
from .deps import KMeans, np


def add_elbow_curve(out: Dict[str, Any], ctx: PlotContext) -> None:
    """Elbow curve for KMeans / MiniBatchKMeans.

    Approximate; fits small k-range on a subset.
    """
    if np is None or KMeans is None:
        return

    if ctx.est_name not in {"KMeans", "MiniBatchKMeans"}:
        return

    try:
        # Use up to ~2000 points for elbow to keep runtime reasonable.
        n_elbow = int(min(ctx.n, 2000))
        Xa_elbow = ctx.Xa
        if ctx.n > n_elbow:
            rng = np.random.default_rng(int(ctx.seed))
            pick = rng.choice(ctx.n, size=n_elbow, replace=False)
            Xa_elbow = ctx.Xa[pick]
            ctx.warnings.append(f"Elbow curve computed on a subsample of {n_elbow} points.")

        base_k = int(getattr(ctx.est, "n_clusters", 2) or 2)
        k_max = int(min(max(6, base_k + 4), 12, max(3, n_elbow - 1)))
        ks = list(range(2, k_max + 1))
        ys = []
        for k in ks:
            try:
                km = KMeans(n_clusters=int(k), random_state=int(ctx.seed), n_init=5, max_iter=200)
                km.fit(Xa_elbow)
                ys.append(float(getattr(km, "inertia_", float("nan"))))
            except Exception:
                ys.append(float("nan"))

        if ks and any(np.isfinite(v) for v in ys):
            out["elbow_curve"] = {
                "x": [int(k) for k in ks],
                "y": [float(v) if np.isfinite(v) else None for v in ys],
            }

    except Exception as e:
        record_error(out, where="reporting.clustering.plots.elbow", exc=e)
        return
