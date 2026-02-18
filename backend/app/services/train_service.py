"""Backend training services.

Segment 12B: the backend becomes a thin adapter layer.

- Validate/construct API request configs in routers.
- Delegate orchestration to the Engine API (engine.api).
- Return JSON-friendly payloads (dict) to be validated by API response models.

Notes
-----
The backend owns HTTP-specific concerns (like progress tracking). All ML
orchestration is delegated to the Engine public API (engine.api).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from engine.api import train_supervised as bl_train_supervised
from engine.api import train_unsupervised as bl_train_unsupervised
from engine.api import run_label_shuffle_baseline_from_cfg as bl_run_label_shuffle

from engine.contracts.run_config import RunConfig
from engine.contracts.unsupervised_configs import UnsupervisedRunConfig

from ..adapters.io_adapter import LoadError
from ..progress.registry import PROGRESS
from ..progress.callback import RegistryProgressCallback

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

    Delegates to :func:`engine.api.train_supervised`.

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
            # PRE-INIT progress so the first poll doesn't 404.
            # The Engine baseline runner will also call progress.init(), which is fine.
            PROGRESS.init(progress_id, total=n_shuffles, label=f"Shuffling 0/{n_shuffles}…")

            # BL-native progress callback (no attribute injection)
            progress_cb = RegistryProgressCallback(progress_id)

            scores = np.asarray(
                bl_run_label_shuffle(cfg, n_shuffles=n_shuffles, progress=progress_cb),
                dtype=float,
            ).ravel()

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
        try:
            progress_id = getattr(cfg.eval, "progress_id", None)
            if progress_id:
                PROGRESS.fail(progress_id, message=f"{type(e).__name__}: {e}")
        except Exception:
            pass

        payload.setdefault("notes", []).append(
            f"Shuffle baseline failed ({type(e).__name__}: {e})."
        )

    return payload


def train_unsupervised(cfg: UnsupervisedRunConfig) -> Dict[str, Any]:
    """Train an unsupervised model.

    Delegates to :func:`engine.api.train_unsupervised`.
    """

    try:
        result = bl_train_unsupervised(cfg)
    except Exception as e:
        if _is_probable_load_error(e):
            raise LoadError(str(e)) from e
        raise

    return to_payload(result)
