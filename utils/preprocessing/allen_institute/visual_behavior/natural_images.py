# utils/preprocessing/allen/natural_images.py
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from ..common import get_dff_and_timestamps, build_windowed_features


def _select_change_detection_block(stim: "pd.DataFrame", substr: str = "change_detection") -> "pd.DataFrame":
    """Filter stimulus_presentations to the change-detection block (VB >=2.16)."""
    mask = stim["stimulus_block_name"].astype(str).str.contains(substr, case=False, na=False)
    out = stim.loc[mask].reset_index(drop=True)
    if len(out) == 0:
        raise RuntimeError(
            f"No rows where stimulus_block_name contains '{substr}'. "
            f"Available: {stim['stimulus_block_name'].unique()}"
        )
    # Normalize end_time naming (older code sometimes expects 'stop_time')
    if "end_time" in out.columns and "stop_time" not in out.columns:
        out = out.rename(columns={"end_time": "stop_time"})
    if "start_time" not in out.columns:
        raise KeyError(f"Missing 'start_time'. Columns: {list(out.columns)}")
    return out.dropna(subset=["start_time"])


def build_natural_image_trials(
    exp,
    *,
    feature: str = "mean",
    pre_window_s: float = 0.5,
    post_window_s: float = 1.0,
    tail_extra_s: float = 0.0,
    min_window_frames: int = 4,
) -> Dict[str, object]:
    """
    Build (X, y) for **natural images** (Visual Behavior, change-detection block).

    Labels default to 'image_name'. If missing, falls back to 'is_change' or
    'stimulus_block_name'.
    """
    roi_ids, dff, t = get_dff_and_timestamps(exp)
    stim_all = exp.stimulus_presentations
    stim = _select_change_detection_block(stim_all)

    # Choose label key by preference
    label_key = None
    if "image_name" in stim.columns and stim["image_name"].notna().any():
        label_key = "image_name"
    elif "is_change" in stim.columns and stim["is_change"].notna().any():
        label_key = "is_change"
    else:
        label_key = "stimulus_block_name"  # fallback

    starts = np.asarray(stim["start_time"], dtype=float)
    X, kept = build_windowed_features(
        dff, t, starts,
        feature=feature,
        pre_window_s=pre_window_s,
        post_window_s=post_window_s,
        tail_extra_s=tail_extra_s,
        min_window_frames=min_window_frames,
    )
    y = stim[label_key].to_numpy()[kept]

    meta = {
        "preprocessor": "natural_images",
        "feature": feature,
        "pre_window_s": pre_window_s,
        "post_window_s": post_window_s,
        "tail_extra_s": tail_extra_s,
        "min_window_frames": min_window_frames,
        "label_key": label_key,
        "n_trials": int(X.shape[0]),
        "n_neurons": int(X.shape[1]),
        "roi_ids": roi_ids.tolist(),
        "kept_trial_indices": kept.tolist(),
    }

    return {"X": X, "y": y, "roi_ids": roi_ids, "meta": meta}
