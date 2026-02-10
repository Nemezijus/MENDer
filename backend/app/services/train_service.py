"""Backend training services.

Segment 12B: the backend becomes a thin adapter layer.

- Validate/construct API request configs in routers.
- Delegate orchestration to the Engine façade.
- Return JSON-friendly payloads (dict) to be validated by API response models.

Notes
-----
The backend still owns HTTP-specific concerns (like progress tracking).
For now, the label-shuffle baseline remains here because it depends on the
backend progress registry.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from engine.use_cases.facade import train_supervised as bl_train_supervised
from engine.use_cases.facade import train_unsupervised as bl_train_unsupervised

from engine.contracts.run_config import RunConfig
from engine.contracts.unsupervised_configs import UnsupervisedRunConfig

from engine.factories.data_loading_factory import make_data_loader
from engine.factories.baseline_factory import make_baseline
from engine.runtime.random.rng import RngManager

from ..adapters.io_adapter import LoadError
from ..progress.registry import PROGRESS

from .common.result_coercion import to_payload


def _is_probable_load_error(exc: Exception) -> bool:
    """Best-effort classification of 'data load' failures.

    The Engine BL does not depend on backend exception types, so data-loading
    errors (FileNotFoundError, missing keys, etc.) would otherwise surface as
    generic 500s.

    This heuristic keeps backward-compatible 400 responses for common load
    failures while not masking genuine training errors.
    """

    if isinstance(exc, (FileNotFoundError, OSError, IOError)):
        return True

    msg = str(exc).lower()
    needles = (
        "npz",
        "npy",
        "mat",
        "h5",
        "hdf5",
        "csv",
        "file",
        "path",
        "no such file",
        "not found",
        "missing",
        "x_key",
        "y_key",
        "load",
        "dataset",
        "could not read",
        "cannot read",
    )
    return any(n in msg for n in needles)


def train(cfg: RunConfig) -> Dict[str, Any]:
    """Train a supervised model.

    Delegates to :func:`engine.use_cases.facade.train_supervised`.

    Returns a JSON-friendly mapping compatible with backend TrainResponse.
    """

    try:
        result = bl_train_supervised(cfg)
    except Exception as e:
        if _is_probable_load_error(e):
            raise LoadError(str(e)) from e
        raise

    payload: Dict[str, Any] = to_payload(result)

    # ------------------------------------------------------------------
    # Optional label-shuffle baseline (backend-owned progress tracking)
    # ------------------------------------------------------------------
    try:
        n_shuffles = int(getattr(cfg.eval, "n_shuffles", 0) or 0)
        progress_id = getattr(cfg.eval, "progress_id", None)

        # Preserve previous behavior: only run when progress_id is provided.
        if n_shuffles > 0 and progress_id:
            # PRE-INIT progress so the first poll doesn't 404
            PROGRESS.init(progress_id, total=n_shuffles, label=f"Shuffling 0/{n_shuffles}…")

            # Load data again (baseline runner needs X,y). This keeps the backend
            # thin while maintaining parity until baseline moves fully into BL.
            loader = make_data_loader(cfg.data)
            X, y = loader.load()

            rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))
            baseline = make_baseline(cfg, rngm)

            # Inject progress registry + parameters for the runner (legacy runner API)
            setattr(baseline, "progress_id", progress_id)
            setattr(baseline, "_progress_total", n_shuffles)
            setattr(baseline, "_progress", PROGRESS)

            scores = np.asarray(baseline.run(X, y), dtype=float).ravel()

            # Compare against main model score
            ref_score = payload.get("mean_score")
            if ref_score is None:
                ref_score = payload.get("metric_value")
            try:
                ref_score_f = float(ref_score)
            except Exception:
                ref_score_f = float("nan")

            ge = int(np.sum(scores >= ref_score_f))
            p_val = (ge + 1.0) / (scores.size + 1.0) if scores.size else float("nan")

            payload["shuffled_scores"] = [float(v) for v in scores.tolist()]
            payload["p_value"] = float(p_val)
            payload.setdefault("notes", []).append(
                f"Shuffle baseline: mean={float(np.mean(scores)):.4f} ± {float(np.std(scores)):.4f}, p≈{float(p_val):.4f}"
            )

    except Exception as e:
        # Baseline errors must not break training.
        payload.setdefault("notes", []).append(
            f"Shuffle baseline failed ({type(e).__name__}: {e})."
        )

    return payload


def train_unsupervised(cfg: UnsupervisedRunConfig) -> Dict[str, Any]:
    """Train an unsupervised model.

    Delegates to :func:`engine.use_cases.facade.train_unsupervised`.
    """

    try:
        result = bl_train_unsupervised(cfg)
    except Exception as e:
        if _is_probable_load_error(e):
            raise LoadError(str(e)) from e
        raise

    return to_payload(result)
