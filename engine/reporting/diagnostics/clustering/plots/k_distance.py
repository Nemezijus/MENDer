from __future__ import annotations

from typing import Any, Dict

from .context import PlotContext
from .deps import NearestNeighbors, np


def add_k_distance_and_dbscan_counts(out: Dict[str, Any], ctx: PlotContext) -> None:
    """DBSCAN helpers: k-distance curve + core/border/noise counts."""
    if np is None or NearestNeighbors is None:
        return

    if ctx.est_name != "DBSCAN":
        return

    try:
        k = int(getattr(ctx.est, "min_samples", 5) or 5)
        k = max(2, k)

        n_kd = int(min(ctx.n, 3000))
        Xa_kd = ctx.Xa
        if ctx.n > n_kd:
            rng = np.random.default_rng(int(ctx.seed))
            pick = rng.choice(ctx.n, size=n_kd, replace=False)
            Xa_kd = ctx.Xa[pick]
            ctx.warnings.append(f"k-distance computed on a subsample of {n_kd} points.")

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(Xa_kd)
        dists, _ = nn.kneighbors(Xa_kd)
        kd = np.sort(dists[:, -1].astype(float))

        out["k_distance"] = {"k": int(k), "y": [float(v) for v in kd.tolist()]}

        core_idx = getattr(ctx.est, "core_sample_indices_", None)
        if core_idx is None:
            return

        core = np.zeros((ctx.n,), dtype=bool)
        try:
            core[np.asarray(core_idx, dtype=int)] = True
            noise = (ctx.y == -1)
            border = (~core) & (~noise)
            out["core_border_noise_counts"] = {
                "core": int(np.sum(core)),
                "border": int(np.sum(border)),
                "noise": int(np.sum(noise)),
            }
        except Exception:
            return

    except Exception:
        return
