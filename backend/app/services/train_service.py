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


from engine.api import train_supervised as bl_train_supervised
from engine.api import train_unsupervised as bl_train_unsupervised
from engine.api import run_label_shuffle_baseline_from_cfg as bl_run_label_shuffle
from engine.api import summarize_label_shuffle_baseline as bl_summarize_label_shuffle_baseline
from engine.api import format_label_shuffle_baseline_failure_note as bl_format_label_shuffle_baseline_failure_note

from engine.contracts.run_config import RunConfig
from engine.contracts.unsupervised_configs import UnsupervisedRunConfig

from ..adapters.io_adapter import LoadError
from ..progress.registry import PROGRESS
from ..progress.callback import RegistryProgressCallback

from .common.error_classification import is_probable_load_error
from .common.result_coercion import to_payload


def train(cfg: RunConfig) -> Dict[str, Any]:
    """Train a supervised model.

    Delegates to :func:`engine.api.train_supervised`.

    Returns a JSON-friendly mapping compatible with backend TrainResponse.
    """

    try:
        result = bl_train_supervised(cfg)
    except Exception as e:
        if is_probable_load_error(e):
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

            scores = bl_run_label_shuffle(cfg, n_shuffles=n_shuffles, progress=progress_cb)

            ref_score = result.mean_score if getattr(result, "mean_score", None) is not None else result.metric_value
            section = bl_summarize_label_shuffle_baseline(scores=list(scores), ref_score=float(ref_score))

            payload["shuffled_scores"] = section.get("shuffled_scores")
            payload["p_value"] = section.get("p_value")
            payload.setdefault("notes", []).extend(section.get("notes", []))
    except Exception as e:
        # Baseline errors must not break training.
        try:
            progress_id = getattr(cfg.eval, "progress_id", None)
            if progress_id:
                PROGRESS.fail(progress_id, message=f"{type(e).__name__}: {e}")
        except Exception:
            pass

        payload.setdefault("notes", []).append(bl_format_label_shuffle_baseline_failure_note(exc_type=type(e).__name__, exc_message=str(e), parens=True))

    return payload


def train_unsupervised(cfg: UnsupervisedRunConfig) -> Dict[str, Any]:
    """Train an unsupervised model.

    Delegates to :func:`engine.api.train_unsupervised`.
    """

    try:
        result = bl_train_unsupervised(cfg)
    except Exception as e:
        if is_probable_load_error(e):
            raise LoadError(str(e)) from e
        raise

    return to_payload(result)
