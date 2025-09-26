# utils/datafetch/allen_institute.py
from __future__ import annotations

import os
import json
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import warnings
import numpy as np

# Silence the pkg_resources deprecation warning coming from AllenSDK internals
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated",
    category=UserWarning,
)

try:
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache,
    )
except Exception as e:
    raise ImportError(
        "AllenSDK (>=2.16) is required. Install with `pip install 'allensdk>=2.16,<3'`.\n"
        f"Import error: {e}"
    )


# =========================
# Public entry point
# =========================

def prepare_experiment_features(
    experiment_id: Union[int, str],
    out_dir: Union[str, os.PathLike] = "data/allen",
    *,
    feature: str = "mean",                # "mean" | "peak" | "auc"
    pre_window_s: float = 0.5,            # seconds before stim onset for baseline
    post_window_s: float = 1.0,           # seconds after stim onset to include (response)
    tail_extra_s: float = 0.0,            # optional extra seconds to catch Ca tail
    min_window_frames: int = 4,           # skip events with too-few frames
    stim_block_name_substr: str = "change_detection",  # which block to use
    save_npz: bool = True,
    cache_dir: Union[str, os.PathLike] = "data/allen_cache",
) -> Dict[str, np.ndarray]:
    """
    High-level orchestrator for Visual Behavior ophys experiments:
      1) connect cache (S3-backed) → 2) load experiment → 3) get dFF + timestamps
      4) select stim presentations (change_detection) → 5) build trial × neuron features
      6) save to .npz (X, y, meta)

    Labels:
      - Uses `image_name` if present; else falls back to `stimulus_name`/`stimulus_block_name`.

    Windows (seconds) are mapped to frame indices via `ophys_timestamps`.

    Returns a dict with arrays and meta.
    """
    exp_id = int(experiment_id)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = _get_vb_cache(cache_dir)
    exp = _load_vb_experiment(cache, exp_id)

    roi_ids, dff, frame_times = _get_dff_and_timestamps(exp)
    stim = _select_stimulus_presentations(exp, block_substr=stim_block_name_substr)

    X, y, kept_idx = _build_trial_features_vb(
        dff=dff,
        frame_times=frame_times,
        stim_table=stim,
        feature=feature,
        pre_window_s=pre_window_s,
        post_window_s=post_window_s,
        tail_extra_s=tail_extra_s,
        min_window_frames=min_window_frames,
    )

    meta = {
        "experiment_id": exp_id,
        "feature": feature,
        "pre_window_s": pre_window_s,
        "post_window_s": post_window_s,
        "tail_extra_s": tail_extra_s,
        "min_window_frames": min_window_frames,
        "stim_block_filter": stim_block_name_substr,
        "n_trials": int(X.shape[0]),
        "n_neurons": int(X.shape[1]),
        "roi_ids": roi_ids.tolist(),
        "kept_trial_indices_in_original_table": kept_idx.tolist(),
        "stim_table_columns": list(stim.columns),
    }

    if save_npz:
        # Try to include a label hint in filename
        label_key = _choose_label_key(stim)
        out_path = out_dir / f"vb_exp_{exp_id}_{label_key}_{feature}.npz"
        np.savez_compressed(out_path, X=X, y=y, roi_ids=roi_ids, meta=json.dumps(meta))
        print(f"[INFO] Saved features to {out_path}")

    return {"X": X, "y": y, "roi_ids": roi_ids, "meta": meta}


# =========================
# Subfunctions (single-responsibility)
# =========================

def _get_vb_cache(cache_dir: Union[str, os.PathLike]) -> VisualBehaviorOphysProjectCache:
    """Create an S3-backed VisualBehavior cache under `cache_dir`."""
    cache_dir = str(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)
    return cache


def _load_vb_experiment(cache: VisualBehaviorOphysProjectCache, exp_id: int):
    """
    Load a VisualBehavior BehaviorOphysExperiment by ophys_experiment_id.
    """
    print(f"[INFO] Loading VisualBehavior ophys experiment {exp_id} …")
    exp = cache.get_behavior_ophys_experiment(exp_id)
    # Sanity check: presence of key tables
    _ = exp.stimulus_presentations
    _ = exp.ophys_timestamps
    return exp


def _get_dff_and_timestamps(exp) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract ΔF/F traces and frame times, robust to VB schema variants.

    Returns
    -------
    roi_ids : (n_cells,) int64
    dff     : (n_cells, n_frames) float32
    t       : (n_frames,) float64 (seconds)
    """
    # Prefer the modern DataFrame API
    if hasattr(exp, "dff_traces"):
        dff_df = exp.dff_traces  # pandas DataFrame
        print(f"[DEBUG] dff_traces columns: {list(dff_df.columns)}")

        # --------- IDs ----------
        if "cell_specimen_id" in dff_df.columns:
            roi_ids = dff_df["cell_specimen_id"].to_numpy(dtype=np.int64)
        elif "cell_roi_id" in dff_df.columns:
            roi_ids = dff_df["cell_roi_id"].to_numpy(dtype=np.int64)
        else:
            # try the experiment’s cell table
            try:
                cst = exp.cell_specimen_table
                if isinstance(cst, dict):
                    # some SDK versions return dict-like
                    ids = cst.get("cell_specimen_id", None)
                    if ids is not None:
                        roi_ids = np.asarray(ids, dtype=np.int64).ravel()
                    else:
                        roi_ids = np.arange(len(dff_df), dtype=np.int64)
                else:
                    # pandas DataFrame
                    if "cell_specimen_id" in cst.columns:
                        roi_ids = cst["cell_specimen_id"].to_numpy(dtype=np.int64)
                    else:
                        roi_ids = np.arange(len(dff_df), dtype=np.int64)
            except Exception:
                roi_ids = np.arange(len(dff_df), dtype=np.int64)

        # --------- Traces ----------
        trace_col = None
        for cand in ("dff", "dff_values", "trace", "fluorescence"):
            if cand in dff_df.columns:
                trace_col = cand
                break
        if trace_col is None:
            # auto-detect: first column whose first element is array-like
            for col in dff_df.columns:
                try:
                    v0 = dff_df[col].iloc[0]
                    if hasattr(v0, "__len__"):
                        trace_col = col
                        break
                except Exception:
                    pass
        if trace_col is None:
            raise RuntimeError(
                "Could not find a ΔF/F column in exp.dff_traces. "
                f"Available columns: {list(dff_df.columns)}"
            )

        dff_list = dff_df[trace_col].to_list()
        dff = np.vstack([np.asarray(x, dtype=np.float32) for x in dff_list])

    else:
        # Legacy tuple API: (ids, dff_array)
        ids, dff_arr = exp.get_dff_traces()
        roi_ids = np.asarray(ids, dtype=np.int64).ravel()
        dff = np.asarray(dff_arr, dtype=np.float32)
        if dff.ndim != 2:
            raise RuntimeError(f"Unexpected dff shape from legacy API: {dff.shape}")

    t = np.asarray(exp.ophys_timestamps, dtype=float)
    print(f"[INFO] dFF shape: {dff.shape} (cells × frames); timestamps: {t.shape}")
    return roi_ids, dff, t




def _select_stimulus_presentations(exp, block_substr: str):
    """
    Filter stim presentations to a specific block (e.g., 'change_detection').
    Accepts either 'end_time' or legacy 'stop_time' columns.
    """
    stim = exp.stimulus_presentations.copy()

    # Per AllenSDK 2.16 warning: pick the 'change_detection' block (or your custom substring)
    mask = stim["stimulus_block_name"].astype(str).str.contains(block_substr, case=False, na=False)
    stim = stim.loc[mask].reset_index(drop=True)

    if len(stim) == 0:
        raise RuntimeError(
            f"No stimulus presentations found containing '{block_substr}' in 'stimulus_block_name'. "
            f"Available block names: {exp.stimulus_presentations['stimulus_block_name'].unique()}"
        )

    # Normalize column names: VB uses 'end_time' (seconds); older code might expect 'stop_time'
    if "end_time" in stim.columns and "stop_time" not in stim.columns:
        stim = stim.rename(columns={"end_time": "stop_time"})

    # We only require start_time; stop_time is optional for our current pipeline
    if "start_time" not in stim.columns:
        raise KeyError(f"'start_time' column is missing. Columns present: {list(stim.columns)}")

    # Drop NaN starts; if 'stop_time' exists, drop NaNs there too
    subset_cols = ["start_time"] + (["stop_time"] if "stop_time" in stim.columns else [])
    stim = stim.dropna(subset=subset_cols)

    return stim



def _choose_label_key(stim) -> str:
    """
    Choose a label column for classification. Preference order:
      1) 'image_name' (multi-class)
      2) 'is_change' (binary, if available)
      3) 'stimulus_block_name' (fallback)
    """
    if "image_name" in stim.columns and stim["image_name"].notna().any():
        return "image_name"
    if "is_change" in stim.columns and stim["is_change"].notna().any():
        return "is_change"
    return "stimulus_block_name"


def _labels_from_stim(stim) -> np.ndarray:
    """Return a 1D array of labels chosen by _choose_label_key."""
    key = _choose_label_key(stim)
    vals = stim[key].to_numpy()
    # Ensure a simple 1D label vector (strings or numbers OK)
    return vals


def _time_to_frame_indices(
    frame_times: np.ndarray, start_s: float, stop_s: float
) -> Tuple[int, int]:
    """
    Map time (s) → nearest frame indices [i0, i1] inclusive bounds.
    """
    i0 = int(np.searchsorted(frame_times, start_s, side="left"))
    i1 = int(np.searchsorted(frame_times, stop_s, side="right")) - 1
    i0 = max(0, min(i0, len(frame_times) - 1))
    i1 = max(0, min(i1, len(frame_times) - 1))
    if i1 < i0:
        i1 = i0
    return i0, i1


def _summarize_segment(seg: np.ndarray, method: str) -> np.ndarray:
    """
    Summarize a (cells × frames) segment into per-cell features.
    """
    if seg.ndim != 2:
        raise ValueError("segment must be 2D (cells × frames)")
    if method == "mean":
        return np.mean(seg, axis=1)
    elif method == "peak":
        return np.max(seg, axis=1)
    elif method == "auc":
        return np.sum(seg, axis=1)  # discrete integral (frame units)
    else:
        raise ValueError(f"Unknown feature method '{method}'")


def _build_trial_features_vb(
    dff: np.ndarray,
    frame_times: np.ndarray,
    stim_table,
    *,
    feature: str = "mean",
    pre_window_s: float = 0.5,
    post_window_s: float = 1.0,
    tail_extra_s: float = 0.0,
    min_window_frames: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build trial × neuron features from Visual Behavior stimulus presentations.

    For each presentation row:
      - Define baseline window: [start_time - pre_window_s, start_time)
      - Define response window: [start_time, start_time + post_window_s + tail_extra_s)
      - Baseline = median over baseline window (per neuron)
      - Feature = summarize (baseline-subtracted response segment) by `feature`

    Returns
    -------
    X : (n_trials_kept, n_neurons) float32
    y : (n_trials_kept,)          object or numeric (labels)
    kept_rows : (n_trials_kept,)  int indices back into `stim_table`
    """
    n_cells, n_frames = dff.shape
    starts_s = np.asarray(stim_table["start_time"], dtype=float)

    # Pick labels (image_name preferred)
    y_full = _labels_from_stim(stim_table)

    feats: List[np.ndarray] = []
    labels: List[Union[str, float, int]] = []
    kept_rows: List[int] = []

    for i, s in enumerate(starts_s):
        # Windows in seconds
        base_s0 = s - float(pre_window_s)
        base_s1 = s
        resp_s0 = s
        resp_s1 = s + float(post_window_s) + float(tail_extra_s)

        # Map to frames
        b0, b1 = _time_to_frame_indices(frame_times, base_s0, base_s1)
        r0, r1 = _time_to_frame_indices(frame_times, resp_s0, resp_s1)

        # Build segments (inclusive indices)
        base_len = b1 - b0 + 1
        resp_len = r1 - r0 + 1
        if base_len <= 0 or resp_len < min_window_frames:
            continue

        base_seg = dff[:, b0 : b1 + 1]
        resp_seg = dff[:, r0 : r1 + 1]

        # Per-neuron baseline (robust median)
        baseline = np.median(base_seg, axis=1)

        # Baseline-subtracted response
        resp_bs = resp_seg - baseline[:, None]

        # Summarize to per-cell feature
        f = _summarize_segment(resp_bs, method=feature).astype(np.float32)

        feats.append(f)
        labels.append(y_full[i])
        kept_rows.append(i)

    if not feats:
        raise RuntimeError(
            "No trials passed the windowing/min_frame filters. "
            "Try reducing pre_window_s/post_window_s or min_window_frames."
        )

    X = np.vstack(feats).astype(np.float32)        # (trials × neurons)
    y = np.asarray(labels)                          # (trials,)
    kept = np.asarray(kept_rows, dtype=int)

    print(f"[INFO] Built features: X {X.shape} (trials × neurons), y {y.shape}")
    return X, y, kept


# =========================
# (Optional) quick manual test
# =========================

if __name__ == "__main__":
    """
    Minimal ad-hoc test:
      python -m utils.datafetch.allen_institute <ophys_experiment_id>
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m utils.datafetch.allen_institute <ophys_experiment_id>")
        sys.exit(1)

    exp_id_cli = sys.argv[1]
    out = prepare_experiment_features(
        exp_id_cli,
        out_dir="data/allen",
        feature="mean",           # try: "peak", "auc"
        pre_window_s=0.5,
        post_window_s=1.0,
        tail_extra_s=0.0,
        min_window_frames=4,
        stim_block_name_substr="change_detection",
        save_npz=True,
        cache_dir="data/allen_cache",
    )
    print(f"[DONE] Trials: {out['X'].shape[0]}, Neurons: {out['X'].shape[1]}")
