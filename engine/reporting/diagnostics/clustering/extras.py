from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from .core_metrics import _as_2d, _final_estimator, _transform_through_pipeline
from .plots import (
    add_centroid_profiles,
    add_cluster_sizes,
    add_decoder_payload,
    add_dendrogram,
    add_distance_to_center,
    add_elbow_curve,
    add_embedding_labels,
    add_gmm_ellipses,
    add_k_distance_and_dbscan_counts,
    add_separation,
    add_silhouette,
    add_spectral_eigenvalues,
)
from .plots.context import PlotContext
from .plots.deps import np


def build_plot_data(
    *,
    model: Any,
    X: Any,
    labels: Any,
    per_sample: Optional[Mapping[str, Any]] = None,
    embedding: Optional[Mapping[str, Any]] = None,
    max_points: int = 5000,
    seed: int = 0,
) -> Tuple[Mapping[str, Any], List[str]]:
    """Build JSON-friendly plot payloads for the frontend.

    Derived values for plots only. Never raises in normal use.

    This is intentionally conservative and best-effort.

    NOTE: This function intentionally mirrors the pre-refactor plot payloads
    so the existing frontend can render the richer diagnostics without
    requiring UI refactors.
    """

    warnings: List[str] = []
    out: Dict[str, Any] = {}

    if np is None:
        return out, ["numpy unavailable; cannot build plot data."]

    # --- validate inputs -------------------------------------------------
    try:
        y = np.asarray(labels).reshape(-1)
    except Exception as e:
        return out, [f"Invalid labels for plot_data: {e}"]

    try:
        Xt = _transform_through_pipeline(model, X)
        Xa = _as_2d(Xt)
    except Exception:
        try:
            Xa = _as_2d(X)
        except Exception as e:
            return out, [f"Invalid inputs for plot_data: {e}"]

    n = int(Xa.shape[0])
    if n == 0:
        return out, ["Empty X; cannot build plot data."]

    if y.size != n:
        m = int(min(y.size, n))
        warnings.append(
            f"Label length ({int(y.size)}) does not match X rows ({int(n)}); using first {m} samples."
        )
        y = y[:m]
        Xa = Xa[:m]
        n = int(m)

    # Downsample indices for payload-heavy arrays
    idx = np.arange(n, dtype=int)
    if int(max_points) > 0 and n > int(max_points):
        try:
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(n, size=int(max_points), replace=False)
            idx = np.sort(idx)
            warnings.append(f"Downsampled plot_data arrays to {int(max_points)} points.")
        except Exception:
            idx = idx[: int(max_points)]
            warnings.append(f"Downsampled plot_data arrays to first {int(max_points)} points.")

    est = _final_estimator(model)
    est_name = type(est).__name__

    ctx = PlotContext(
        model=model,
        est=est,
        est_name=est_name,
        Xa=Xa,
        y=y,
        n=n,
        idx=idx,
        per_sample=per_sample,
        embedding=embedding,
        seed=int(seed),
        warnings=warnings,
    )

    # ------------------------------------------------------------------
    # Global/common plots
    # ------------------------------------------------------------------

    add_cluster_sizes(out, ctx)
    add_distance_to_center(out, ctx)
    add_centroid_profiles(out, ctx)
    add_separation(out, ctx)
    add_silhouette(out, ctx)
    add_decoder_payload(out, ctx)

    # ------------------------------------------------------------------
    # Model-specific plots
    # ------------------------------------------------------------------

    add_elbow_curve(out, ctx)
    add_k_distance_and_dbscan_counts(out, ctx)
    add_spectral_eigenvalues(out, ctx)
    add_dendrogram(out, ctx)

    # Embedding overlays
    add_embedding_labels(out, ctx)
    add_gmm_ellipses(out, ctx)

    return out, warnings
