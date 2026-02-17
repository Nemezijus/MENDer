from __future__ import annotations

from typing import Any, Dict

from .context import PlotContext
from .deps import np


def add_spectral_eigenvalues(out: Dict[str, Any], ctx: PlotContext) -> None:
    """Spectral clustering eigenvalue spectrum (optional; small n only)."""
    if np is None:
        return

    if ctx.est_name != "SpectralClustering":
        return

    try:
        A = getattr(ctx.est, "affinity_matrix_", None)
        if A is None:
            return

        try:
            if hasattr(A, "toarray"):
                A = A.toarray()
        except Exception:
            pass

        A = np.asarray(A, dtype=float)
        nA = int(A.shape[0])
        if A.ndim != 2 or A.shape[0] != A.shape[1] or nA < 2:
            return

        if nA > 400:
            ctx.warnings.append(f"Spectral eigenvalues skipped: affinity matrix too large ({nA}x{nA}).")
            return

        A = 0.5 * (A + A.T)
        d = np.sum(A, axis=1)
        d = np.maximum(d, 1e-12)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        L = np.eye(nA) - (D_inv_sqrt @ A @ D_inv_sqrt)
        w = np.linalg.eigvalsh(L)
        w = np.sort(np.asarray(w, dtype=float))
        m_keep = int(min(25, w.size))
        out["spectral_eigenvalues"] = {"values": [float(v) for v in w[:m_keep].tolist()]}

    except Exception:
        return
