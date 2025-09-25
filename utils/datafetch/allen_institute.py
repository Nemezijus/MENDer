# utils/datafetch/allen_institute.py
from __future__ import annotations

import os
import json
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# AllenSDK (Visual Coding 2p is served via BrainObservatoryCache - legacy but still supported)
try:
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
except Exception as e:
    raise ImportError(
        "AllenSDK is required. Install with `pip install allensdk`. "
        "Import error: {}".format(e)
    )


# =========================
# Public entry point
# =========================

def prepare_experiment_features(
    experiment_id: Union[int, str],
    out_dir: Union[str, os.PathLike] = "data/allen",
    *,
    stim_preference: Optional[List[str]] = None,
    feature: str = "mean",                # "mean" | "peak" | "auc"
    baseline_frames: int = 15,            # frames before stim onset to compute baseline
    response_extra_frames: int = 0,       # add frames beyond stim end (to catch Ca tail)
    min_trial_frames: int = 5,            # discard very short presentations
    save_npz: bool = True,
) -> Dict[str, np.ndarray]:
    """
    High-level orchestrator:
      1) connect/cache (manifest) → 2) download experiment → 3) extract dFF + stim
      4) pick a gratings stimulus table → 5) build trial × neuron features
      6) save to .npz (X, y, meta)

    Returns a dict with arrays and meta for immediate use.
    """
    exp_id = int(experiment_id)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    boc = _get_boc(manifest_path=out_dir / "manifest.json")
    exp = _download_experiment(boc, exp_id)

    roi_ids, dff = _extract_dff_traces(exp)  # dff shape: (n_cells, n_frames)

    stimuli_available = _list_stimuli(exp)
    stim_name = _choose_stimulus(stimuli_available, stim_preference)

    stim_table = _get_stimulus_table(exp, stim_name)
    X, y, kept_rows = _build_trial_features_from_dff(
        dff=dff,
        stim_table=stim_table,
        feature=feature,
        baseline_frames=baseline_frames,
        response_extra_frames=response_extra_frames,
        min_trial_frames=min_trial_frames,
    )

    meta = {
        "experiment_id": exp_id,
        "stimulus_used": stim_name,
        "feature": feature,
        "baseline_frames": baseline_frames,
        "response_extra_frames": response_extra_frames,
        "min_trial_frames": min_trial_frames,
        "n_trials": int(X.shape[0]),
        "n_neurons": int(X.shape[1]),
        "roi_ids": roi_ids,
        "kept_trial_indices_in_original_table": kept_rows,
        "stim_table_columns": list(stim_table.columns) if hasattr(stim_table, "columns") else None,
    }

    if save_npz:
        out_path = out_dir / f"exp_{exp_id}_{stim_name}_{feature}.npz"
        np.savez_compressed(out_path, X=X, y=y, roi_ids=roi_ids, meta=json.dumps(meta))
        print(f"[INFO] Saved features to {out_path}")

    return {"X": X, "y": y, "roi_ids": roi_ids, "meta": meta}


# =========================
# Subfunctions (single-responsibility)
# =========================

def _get_boc(manifest_path: Union[str, os.PathLike]) -> BrainObservatoryCache:
    """
    Create (or reuse) a BrainObservatoryCache with manifest on disk.
    """
    manifest_path = str(manifest_path)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    boc = BrainObservatoryCache(manifest_file=manifest_path)
    return boc


def _download_experiment(boc: BrainObservatoryCache, experiment_id: int):
    """
    Download and return an Experiment object for the given experiment ID.
    """
    print(f"[INFO] Fetching experiment {experiment_id} …")
    exp = boc.get_experiment_data(experiment_id)
    return exp


def _extract_dff_traces(exp) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ΔF/F traces (cells × frames) and ROI IDs.

    Returns
    -------
    roi_ids : np.ndarray of shape (n_cells,)
    dff     : np.ndarray of shape (n_cells, n_frames)
    """
    # AllenSDK provides:
    #   dff, roi_ids = exp.get_dff_traces()
    dff, roi_ids = exp.get_dff_traces()
    dff = np.asarray(dff, dtype=np.float32)      # (n_cells, n_frames)
    roi_ids = np.asarray(roi_ids, dtype=np.int64)
    print(f"[INFO] dFF shape: {dff.shape} (cells × frames)")
    return roi_ids, dff


def _list_stimuli(exp) -> List[str]:
    """
    Return a list of stimulus names available in this experiment.
    """
    names = exp.list_stimuli()
    print(f"[INFO] Stimuli available: {names}")
    return names


def _choose_stimulus(available: List[str], preference: Optional[List[str]]) -> str:
    """
    Pick a stimulus name, preferring gratings if present (contrast decoding).
    """
    if preference:
        for name in preference:
            if name in available:
                print(f"[INFO] Using preferred stimulus: {name}")
                return name

    # Heuristic: drifting_gratings > static_gratings > anything else
    for candidate in ("drifting_gratings", "static_gratings"):
        if candidate in available:
            print(f"[INFO] Using stimulus: {candidate}")
            return candidate

    # Fallback to first available
    chosen = available[0]
    print(f"[WARN] No gratings found; falling back to: {chosen}")
    return chosen


def _get_stimulus_table(exp, stim_name: str):
    """
    Get the stimulus table (pandas DataFrame-like) for a chosen stimulus.
    Should contain 'start'/'end' frame indices and (for gratings) 'contrast'.
    """
    stim_table = exp.get_stimulus_table(stim_name)
    # Sanity checks
    for col in ("start", "end"):
        if col not in stim_table.columns:
            raise ValueError(f"Stimulus table missing required column '{col}'.")
    if "contrast" not in stim_table.columns:
        print("[WARN] No 'contrast' column found; labels will be uniform or absent.")
    print(f"[INFO] Stim table rows: {len(stim_table)}; columns: {list(stim_table.columns)}")
    return stim_table


def _baseline_per_trial(
    dff: np.ndarray, start_frame: int, baseline_frames: int
) -> np.ndarray:
    """
    Compute per-neuron baseline (median) for frames [start_frame - baseline_frames, start_frame).
    Clips window to start at frame 0 if needed.

    Returns: (n_cells,) baseline vector.
    """
    n_cells, n_frames = dff.shape
    a = max(0, int(start_frame) - int(baseline_frames))
    b = max(0, int(start_frame))
    if b <= a:
        # no room for baseline; return zeros to avoid shifting
        return np.zeros((n_cells,), dtype=np.float32)
    baseline = np.median(dff[:, a:b], axis=1)
    return baseline.astype(np.float32)


def _summarize_response_window(
    segment: np.ndarray, method: str
) -> np.ndarray:
    """
    Summarize a (n_cells × k_frames) segment into a (n_cells,) vector.

    method: "mean" | "peak" | "auc"
    """
    if segment.ndim != 2:
        raise ValueError("segment must be 2D (cells × frames)")
    if method == "mean":
        return np.mean(segment, axis=1)
    elif method == "peak":
        return np.max(segment, axis=1)
    elif method == "auc":
        # simple discrete integral (sum over frames)
        return np.sum(segment, axis=1)
    else:
        raise ValueError(f"Unknown feature method '{method}'")


def _build_trial_features_from_dff(
    dff: np.ndarray,
    stim_table,
    *,
    feature: str = "mean",
    baseline_frames: int = 15,
    response_extra_frames: int = 0,
    min_trial_frames: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct trial × neuron features and labels from dFF and a stimulus table.

    For each row in stim_table:
      - Defines [start, end] in frames (plus optional tail frames).
      - Computes per-neuron baseline from pre-stim frames.
      - Baseline-subtracts the response window.
      - Summarizes by 'feature' (mean/peak/auc).
      - Label = 'contrast' if present; else zeros.

    Returns
    -------
    X : (n_trials_kept, n_neurons) float32
    y : (n_trials_kept,) float32 or float64 (contrast or fallback label)
    kept_rows : (n_trials_kept,) indices of rows kept from original stim_table
    """
    n_cells, n_frames = dff.shape
    starts = np.asarray(stim_table["start"]).astype(int)
    ends = np.asarray(stim_table["end"]).astype(int)

    # Labels: contrast if available; otherwise zeros
    if "contrast" in stim_table.columns:
        y_full = np.asarray(stim_table["contrast"]).astype(float)
    else:
        y_full = np.zeros_like(starts, dtype=float)

    rows: List[int] = []
    feats: List[np.ndarray] = []
    labels: List[float] = []

    for i, (s, e) in enumerate(zip(starts, ends)):
        # Expand response window to include tail (if any)
        e_expanded = min(e + int(response_extra_frames), n_frames - 1)
        k = e_expanded - s + 1
        if k < min_trial_frames:
            continue  # too short; skip

        # Compute baseline over pre-stim window
        b = _baseline_per_trial(dff, s, baseline_frames)

        # Extract response segment and baseline-subtract
        seg = dff[:, s : e_expanded + 1] - b[:, None]   # (cells × k)

        # Summarize to 1 value per neuron
        f = _summarize_response_window(seg, feature=feature)  # (cells,)
        feats.append(f.astype(np.float32))
        labels.append(float(y_full[i]))
        rows.append(i)

    if not feats:
        raise RuntimeError("No trials passed the filters; try reducing min_trial_frames.")

    X = np.vstack(feats)               # (trials × neurons) currently (n, cells)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(labels)

    # We expect (trials × neurons); most of your pipeline expects that
    # so keep as-is. If you want neurons × trials, transpose outside.
    print(f"[INFO] Built features: X {X.shape} (trials × neurons), y {y.shape}")
    return X, y, np.asarray(rows, dtype=int)


# =========================
# (Optional) quick manual test
# =========================

if __name__ == "__main__":
    """
    Minimal ad-hoc test:
      python -m utils.datafetch.allen_institute 511509529
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m utils.datafetch.allen_institute <experiment_id>")
        sys.exit(1)

    exp_id_cli = sys.argv[1]
    out = prepare_experiment_features(
        exp_id_cli,
        out_dir="data/allen",
        stim_preference=["drifting_gratings", "static_gratings"],
        feature="mean",                # try: "peak", "auc"
        baseline_frames=15,            # ~1–2 s depending on frame rate
        response_extra_frames=0,       # add 0–10 frames to catch tail if desired
        min_trial_frames=5,
        save_npz=True,
    )
    print(f"[DONE] Trials: {out['X'].shape[0]}, Neurons: {out['X'].shape[1]}")
