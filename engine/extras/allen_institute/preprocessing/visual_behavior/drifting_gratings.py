from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from .common import get_dff_and_timestamps, build_windowed_features


def _select_drifting_gratings_rows(stim: "pd.DataFrame") -> "pd.DataFrame":
    """
    Try to select rows corresponding to drifting-gratings. Works if the session actually
    contains them (some Visual Behavior sessions won’t).

    Heuristics (robust across schema):
      - stimulus_name or stimulus_block_name contains 'drifting' OR
      - table has columns like 'orientation' or 'direction'
    """
    df = stim.copy()
    mask = (
        df.get("stimulus_name", "").astype(str).str.contains("drifting", case=False, na=False)
        | df.get("stimulus_block_name", "").astype(str).str.contains("drifting", case=False, na=False)
    )
    if mask.any():
        return df.loc[mask].reset_index(drop=True)

    # If no explicit 'drifting' strings, try presence of orientation/direction columns
    cols = set(df.columns)
    if {"orientation", "direction"} & cols:
        # Keep rows where these values are finite
        keep = np.ones(len(df), dtype=bool)
        for c in ["orientation", "direction"]:
            if c in df.columns:
                keep &= np.isfinite(df[c].to_numpy(dtype=float, copy=False), where=True, out=np.ones(len(df), bool))
        out = df.loc[keep].reset_index(drop=True)
        if len(out):
            return out

    raise RuntimeError(
        "Could not find drifting-gratings presentations in this experiment. "
        "This VB session may not include drifting gratings."
    )


def build_drifting_grating_trials(
    exp,
    *,
    feature: str = "mean",
    pre_window_s: float = 0.5,
    post_window_s: float = 1.0,
    tail_extra_s: float = 0.0,
    min_window_frames: int = 4,
    label_by: str = "orientation",      # 'orientation' | 'direction'
    collapse_orientation: bool = True,  # if labeling by 'direction', set True to mod 180
) -> Dict[str, object]:
    """
    Build (X, y) for *drifting gratings*, if present in this Visual Behavior experiment.

    Notes
    -----
    - Many VB sessions do not contain drifting gratings. In that case this function raises.
    - If present, labels can be either 'orientation' (0..180) or 'direction' (0..360).
      If 'collapse_orientation' is True and label_by == 'direction', labels are folded
      modulo 180 (i.e., directions 0 and 180 become the same orientation class).
    """
    roi_ids, dff, t = get_dff_and_timestamps(exp)
    stim_all = exp.stimulus_presentations
    dg = _select_drifting_gratings_rows(stim_all)

    # Normalize end_time naming
    if "end_time" in dg.columns and "stop_time" not in dg.columns:
        dg = dg.rename(columns={"end_time": "stop_time"})
    if "start_time" not in dg.columns:
        raise KeyError(f"Missing 'start_time'. Columns: {list(dg.columns)}")

    if label_by not in dg.columns:
        # Soft fallback: if labeling by orientation but only direction is present
        if label_by == "orientation" and "direction" in dg.columns:
            label_vals = np.asarray(dg["direction"], dtype=float) % 180.0
            y = label_vals
        else:
            raise KeyError(f"Label column '{label_by}' not in drifting-gratings table: {list(dg.columns)}")
    else:
        label_vals = np.asarray(dg[label_by], dtype=float)
        y = label_vals

    if label_by == "direction" and collapse_orientation:
        # fold to orientation (mod 180)
        y = np.mod(y, 180.0)

    starts = np.asarray(dg["start_time"], dtype=float)
    X, kept = build_windowed_features(
        dff, t, starts,
        feature=feature,
        pre_window_s=pre_window_s,
        post_window_s=post_window_s,
        tail_extra_s=tail_extra_s,
        min_window_frames=min_window_frames,
    )
    y = y[kept]

    meta = {
        "preprocessor": "drifting_gratings",
        "feature": feature,
        "pre_window_s": pre_window_s,
        "post_window_s": post_window_s,
        "tail_extra_s": tail_extra_s,
        "min_window_frames": min_window_frames,
        "label_key": label_by + ("_collapsed180" if (label_by == "direction" and collapse_orientation) else ""),
        "n_trials": int(X.shape[0]),
        "n_neurons": int(X.shape[1]),
        "roi_ids": roi_ids.tolist(),
        "kept_trial_indices": kept.tolist(),
    }

    return {"X": X, "y": y, "roi_ids": roi_ids, "meta": meta}
