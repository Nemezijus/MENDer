# utils/preprocessing/allen_vc/drifting_gratings.py
from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np

def _get_dff_and_framerate(ds) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    VC-2P: ds.get_dff_traces() -> (cell_ids, dff) with (cells x frames).
    Get the ophys frame rate from metadata (or a dedicated getter, if present).
    """
    cell_ids, dff = ds.get_dff_traces()
    dff = np.asarray(dff, dtype=np.float32)  # (cells x frames)

    # Try a few ways to obtain frame rate:
    fr = None
    if hasattr(ds, "get_ophys_frame_rate"):
        try:
            fr = float(ds.get_ophys_frame_rate())
        except Exception:
            fr = None
    if fr is None and hasattr(ds, "get_metadata"):
        try:
            meta = ds.get_metadata()
            if "ophys_frame_rate" in meta:
                fr = float(meta["ophys_frame_rate"])
        except Exception:
            fr = None
    if fr is None:
        # Fallback: many VC sessions are 30 Hz, but warn the user
        fr = 30.0
        print("[WARN] Could not read frame rate from dataset; assuming 30.0 Hz.")

    return np.asarray(cell_ids, dtype=np.int64), dff, fr

def _stim_table_dg(ds):
    """
    VC-2P: get table for drifting gratings.
    Columns typically include: 'start', 'end', 'orientation', 'TF', 'SF', 'direction'.
    """
    stim = ds.get_stimulus_table('drifting_gratings')  # raises if absent
    # normalize columns
    cols = {c.lower(): c for c in stim.columns}
    # Required
    start_col = cols.get('start'); end_col = cols.get('end')
    if start_col is None:
        raise KeyError(f"'start' not in DG stim table columns: {list(stim.columns)}")
    return stim, start_col, end_col

def _time_to_frame_idx(t: np.ndarray, a: float, b: float) -> Tuple[int, int]:
    i0 = int(np.searchsorted(t, a, side="left"))
    i1 = int(np.searchsorted(t, b, side="right")) - 1
    i0 = max(0, min(i0, len(t)-1))
    i1 = max(0, min(i1, len(t)-1))
    if i1 < i0: i1 = i0
    return i0, i1

def _summarize(seg: np.ndarray, method: str) -> np.ndarray:
    if method == "mean": return np.mean(seg, axis=1)
    if method == "peak": return np.max(seg, axis=1)
    if method == "auc":  return np.sum(seg, axis=1)
    raise ValueError(f"Unknown feature method '{method}'")

def build_vc_drifting_gratings_trials(
    ds,
    *,
    feature: str = "mean",
    pre_window_s: float = 0.5,
    post_window_s: float = 1.0,
    tail_extra_s: float = 0.0,
    label_by: str = "orientation",      # 'orientation' | 'direction'
    collapse_orientation: bool = True,  # fold direction mod 180 -> orientation
    bin_orientations: bool = False,     # optional: round to bins (e.g., 45°)
    bin_size_deg: float = 45.0,
) -> Dict[str, object]:
    roi_ids, dff, frame_rate = _get_dff_and_framerate(ds)
    stim, start_col, end_col = _stim_table_dg(ds)

    # ---- labels (build raw float labels, then clean) ----
    if label_by not in {c.lower(): c for c in stim.columns} and label_by not in stim.columns:
        # convenience: map case-insensitive
        for c in stim.columns:
            if c.lower() == label_by.lower():
                label_col = c
                break
        else:
            if label_by == "orientation" and "direction" in stim.columns:
                label_vals = np.mod(stim["direction"].to_numpy(float), 180.0)
            else:
                raise KeyError(f"Label column '{label_by}' not in {list(stim.columns)}")
    else:
        # exact or case-insensitive match
        label_col = next((c for c in stim.columns if c.lower() == label_by.lower()), label_by)
        label_vals = stim[label_col].to_numpy(float)

    if label_by == "direction" and collapse_orientation:
        label_vals = np.mod(label_vals, 180.0)

    # Optionally bin orientations (e.g., to 8 classes at 0,45,…,315)
    if bin_orientations:
        # robust binning: map to nearest multiple of bin_size_deg
        label_vals = (np.round(label_vals / bin_size_deg) * bin_size_deg) % (180.0 if collapse_orientation or label_by=="orientation" else 360.0)

    # ---- drop rows with invalid labels or starts ----
    starts = stim[start_col].to_numpy()  # frame indices
    valid = np.isfinite(label_vals) & np.isfinite(starts)
    if not np.any(valid):
        raise RuntimeError("No drifting-gratings rows with finite labels and starts.")
    label_vals = label_vals[valid]
    starts = starts[valid]
    stim = stim.iloc[valid].reset_index(drop=True)

    # enforce integer frame indices
    starts = starts.astype(int, copy=False)

    # ---- windowing in frames ----
    pre_n  = int(round(pre_window_s * frame_rate))
    post_n = int(round((post_window_s + tail_extra_s) * frame_rate))
    n_cells, n_frames = dff.shape

    feats, kept = [], []
    for i, s in enumerate(starts):
        b0 = max(0, s - pre_n); b1 = max(b0, s)      # [b0, s)
        r0 = s; r1 = min(n_frames, s + post_n)       # [s, r1)
        if (b1 - b0) <= 0 or (r1 - r0) < 3:
            continue
        base = np.median(dff[:, b0:b1], axis=1)
        resp = dff[:, r0:r1] - base[:, None]
        feats.append(_summarize(resp, feature).astype(np.float32))
        kept.append(i)

    if not feats:
        raise RuntimeError("No DG trials passed windowing; try larger post_window_s or smaller pre_window_s.")

    X = np.vstack(feats).astype(np.float32)
    y = label_vals[np.asarray(kept, dtype=int)]

    meta = {
        "preprocessor": "vc_drifting_gratings",
        "feature": feature,
        "pre_window_s": pre_window_s,
        "post_window_s": post_window_s,
        "tail_extra_s": tail_extra_s,
        "label_key": (
            (label_by + "_collapsed180") if (label_by == "direction" and collapse_orientation) else label_by
        ) + (f"_binned_{int(bin_size_deg)}" if bin_orientations else ""),
        "n_trials": int(X.shape[0]),
        "n_neurons": int(X.shape[1]),
        "roi_ids": roi_ids.tolist(),
        "kept_trial_indices": np.asarray(kept, int).tolist(),
        "frame_rate_hz": float(frame_rate),
    }
    return {"X": X, "y": y, "roi_ids": roi_ids, "meta": meta}
