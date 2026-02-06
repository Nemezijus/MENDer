# utils/preprocessing/allen/common.py
from __future__ import annotations

from typing import Tuple, List, Dict, Union
import json
import numpy as np


def get_dff_and_timestamps(exp) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract ΔF/F traces and frame times, robust across VB schema variants.

    Returns
    -------
    roi_ids : (n_cells,) int64
    dff     : (n_cells, n_frames) float32
    t       : (n_frames,) float64 (seconds)
    """
    if hasattr(exp, "dff_traces"):
        dff_df = exp.dff_traces  # pandas DataFrame
        # IDs
        if "cell_specimen_id" in dff_df.columns:
            roi_ids = dff_df["cell_specimen_id"].to_numpy(dtype=np.int64)
        elif "cell_roi_id" in dff_df.columns:
            roi_ids = dff_df["cell_roi_id"].to_numpy(dtype=np.int64)
        else:
            # fallback: 0..N-1
            roi_ids = np.arange(len(dff_df), dtype=np.int64)
        # Traces column
        trace_col = None
        for cand in ("dff", "dff_values", "trace", "fluorescence"):
            if cand in dff_df.columns:
                trace_col = cand
                break
        if trace_col is None:
            # auto-detect first array-like column
            for col in dff_df.columns:
                v0 = dff_df[col].iloc[0]
                if hasattr(v0, "__len__"):
                    trace_col = col
                    break
        if trace_col is None:
            raise RuntimeError(f"Could not find dF/F column in {list(dff_df.columns)}")
        dff_list = dff_df[trace_col].to_list()
        dff = np.vstack([np.asarray(x, dtype=np.float32) for x in dff_list])
    else:
        # Legacy tuple API
        ids, dff_arr = exp.get_dff_traces()
        roi_ids = np.asarray(ids, dtype=np.int64).ravel()
        dff = np.asarray(dff_arr, dtype=np.float32)
        if dff.ndim != 2:
            raise RuntimeError(f"Unexpected dff shape: {dff.shape}")

    t = np.asarray(exp.ophys_timestamps, dtype=float)
    return roi_ids, dff, t


def time_to_frame_indices(
    frame_times: np.ndarray, start_s: float, stop_s: float
) -> Tuple[int, int]:
    """Map time (s) → nearest inclusive frame indices [i0, i1]."""
    i0 = int(np.searchsorted(frame_times, start_s, side="left"))
    i1 = int(np.searchsorted(frame_times, stop_s, side="right")) - 1
    i0 = max(0, min(i0, len(frame_times) - 1))
    i1 = max(0, min(i1, len(frame_times) - 1))
    if i1 < i0:
        i1 = i0
    return i0, i1


def summarize_segment(seg: np.ndarray, method: str) -> np.ndarray:
    """Summarize a (cells × frames) segment into per-cell features."""
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


def build_windowed_features(
    dff: np.ndarray,
    frame_times: np.ndarray,
    starts_s: np.ndarray,
    *,
    feature: str = "mean",
    pre_window_s: float = 0.5,
    post_window_s: float = 1.0,
    tail_extra_s: float = 0.0,
    min_window_frames: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core feature builder used by multiple preprocessors.

    For each start time:
      - baseline window: [start - pre_window_s, start)
      - response window: [start, start + post_window_s + tail_extra_s)
      - per-neuron baseline = median(baseline window)
      - features = summarize(baseline-subtracted response) by `feature`
    """
    n_cells, _ = dff.shape
    feats: List[np.ndarray] = []
    kept_idx: List[int] = []

    for i, s in enumerate(starts_s):
        base_s0, base_s1 = s - float(pre_window_s), s
        resp_s0, resp_s1 = s, s + float(post_window_s) + float(tail_extra_s)

        b0, b1 = time_to_frame_indices(frame_times, base_s0, base_s1)
        r0, r1 = time_to_frame_indices(frame_times, resp_s0, resp_s1)

        base_len, resp_len = (b1 - b0 + 1), (r1 - r0 + 1)
        if base_len <= 0 or resp_len < min_window_frames:
            continue

        base_seg = dff[:, b0 : b1 + 1]
        resp_seg = dff[:, r0 : r1 + 1]

        baseline = np.median(base_seg, axis=1)                # (cells,)
        resp_bs = resp_seg - baseline[:, None]                # baseline-subtracted
        f = summarize_segment(resp_bs, method=feature).astype(np.float32)
        feats.append(f)
        kept_idx.append(i)

    if not feats:
        raise RuntimeError(
            "No trials passed the windowing/min_frame filters. "
            "Try adjusting pre/post windows or min_window_frames."
        )

    X = np.vstack(feats).astype(np.float32)
    kept = np.asarray(kept_idx, dtype=int)
    return X, kept

def save_trial_dataset_npz(out: Dict[str, object], path: str) -> str:
    """Save {'X','y','roi_ids','meta'} dict to a compressed NPZ."""
    X = out["X"]; y = out["y"]; roi_ids = out["roi_ids"]; meta = out["meta"]
    np.savez_compressed(path, X=X, y=y, roi_ids=roi_ids, meta=json.dumps(meta))
    print(f"[INFO] Saved dataset to {path}  --  X{X.shape}, y{y.shape}")
    return path