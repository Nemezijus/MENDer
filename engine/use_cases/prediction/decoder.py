from __future__ import annotations

from typing import Any, Optional

import numpy as np

from engine.components.prediction import predict_decoder_outputs


def maybe_compute_decoder_outputs(
    *,
    task: str,
    eval_model: Any,
    pipeline: Any,
    X_arr: np.ndarray,
    y_true: Optional[np.ndarray],
    n_samples: int,
    preview_rows_cap: int,
    notes: list[str],
) -> Any:
    """Compute decoder outputs (classification only) if enabled in eval config."""

    if task != "classification" or eval_model is None:
        return None

    decoder_cfg = getattr(eval_model, "decoder", None)
    decoder_enabled = (
        bool(getattr(decoder_cfg, "enabled", False)) if decoder_cfg is not None else False
    )
    if not decoder_enabled:
        return None

    decoder_positive_label = getattr(decoder_cfg, "positive_class_label", None)
    decoder_include_scores = bool(getattr(decoder_cfg, "include_decision_scores", True))
    decoder_include_probabilities = bool(getattr(decoder_cfg, "include_probabilities", True))
    decoder_include_margin = bool(getattr(decoder_cfg, "include_margin", True))
    decoder_calibrate_probabilities = bool(getattr(decoder_cfg, "calibrate_probabilities", False))

    decoder_preview_cap = int(getattr(decoder_cfg, "max_preview_rows", preview_rows_cap) or preview_rows_cap)
    n_preview_dec = min(decoder_preview_cap, n_samples)

    try:
        return predict_decoder_outputs(
            pipeline,
            X_arr,
            y_true=y_true,
            indices=range(n_samples),
            positive_class_label=decoder_positive_label,
            include_decision_scores=decoder_include_scores,
            include_probabilities=decoder_include_probabilities,
            include_margin=decoder_include_margin,
            calibrate_probabilities=decoder_calibrate_probabilities,
            max_preview_rows=n_preview_dec,
            include_summary=False,
        )
    except Exception as e:
        notes.append(f"Decoder outputs could not be computed ({type(e).__name__}: {e})")
        return None
