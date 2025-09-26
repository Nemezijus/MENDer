"""
Batch runner for ridge_shuffle_decoding

- Loops over multiple y_path files (all in the same directory).
- For each y_path, loops over a list of thresholds (thr).
- Collects Real R^2, Shuffled mean, and Shuffled std into a CSV.

USAGE:
    python batch_ridge_shuffle.py

Customize the CONFIG section below for your paths and parameter defaults.

This script *suppresses plotting* without modifying your model file:
- Forces a non-interactive backend (Agg)
- Temporarily patches plt.show() to a no-op
- Closes any figures after each run
"""

import os
import csv
from pathlib import Path
import numpy as np
import math

# --- (A) Keep GUI off to avoid blocking on plt.show()
os.environ.setdefault("MPLBACKEND", "Agg")

# Import after setting backend
import matplotlib.pyplot as plt

# Try to import your module either by its package path or local file.
# If your project uses `python -m instances.ridge_shuffle_decoding` normally,
# then this import should match that package path:
try:
    from instances.ridge_shuffle_decoding import run_ridge_decoding
except ImportError:
    # Fallback: same file placed alongside this script
    from ridge_shuffle_decoding import run_ridge_decoding


# =========================
# ======= CONFIGURE =======
# =========================

# Single X path fixed across runs
X_PATH = r"E:\work\projects\freely_moving\animals\m29\trial76\combined\calcium\dff.mat"

# Directory of Y .mat files
Y_DIR = Path(r"E:\work\projects\freely_moving\animals\m29\trial76\roboticArmData\splitAndDownsampled")

# List your Y filenames here (no wildcards, explicit control)
Y_FILES = [
    "actual_TCP_force_0.mat",
    "actual_TCP_force_1.mat",
    "actual_TCP_force_2.mat",
    "actual_TCP_force_3.mat",
    "actual_TCP_force_4.mat",
    "actual_TCP_force_5.mat",
    "actual_TCP_pose_0.mat",
    "actual_TCP_pose_1.mat",
    "actual_TCP_pose_2.mat",
    "actual_TCP_pose_3.mat",
    "actual_TCP_pose_4.mat",
    "actual_TCP_pose_5.mat",
    "actual_TCP_speed_0.mat",
    "actual_TCP_speed_1.mat",
    "actual_TCP_speed_2.mat",
    "actual_TCP_speed_3.mat",
    "actual_TCP_speed_4.mat",
    "actual_TCP_speed_5.mat",
    "actual_tool_accelerometer_0.mat",
    "actual_tool_accelerometer_1.mat",
    "actual_tool_accelerometer_2.mat",
    "pitch_deg.mat",
    "roll_deg.mat",
    "yaw_deg.mat",
    "v2d.mat",
    "v3d.mat",
    # add more...
]

# Per-file thresholds (if a file is not listed here, DEFAULT_THR_LIST will be used)
PER_FILE_THR = {
    "actual_TCP_speed_1.mat": [0.0, 0.1, 0.2],
    "actual_TCP_speed_2.mat": [0.0, 0.1, 0.2],
    # "actual_TCP_speed_3.mat": [0.0, 0.1, 0.2, 0.3],
}

DEFAULT_THR_LIST = [0.0, 0.1, 0.2]

# Common decoding parameters (match your CLI example)
N_SHUFFLES = 500
TRAIN_FRAC = 0.75
GAP = 5
ALPHA = 2.0
SEED = 42
USE_FILTER = True        # you used --use_filter
CORR_METHOD = "pearson"  # current code supports only "pearson"

# Output summary CSV
OUT_CSV = Path("./ridge_shuffle_batch_summary.csv")


def permutation_significance(real_score, shuffled_scores):
    arr = np.asarray(shuffled_scores, dtype=float)
    n = arr.size
    mean = float(np.mean(arr)) if n else math.nan
    std  = float(np.std(arr))  if n else math.nan
    # Empirical one-sided p-value with +1 smoothing
    ge = int(np.sum(arr >= real_score)) if n else 0
    p_value = (ge + 1.0) / (n + 1.0) if n else math.nan
    # z-score vs. shuffle distribution (normal approx)
    if n and std > 0:
        z = (real_score - mean) / std
    else:
        z = math.nan
    return mean, std, p_value, z

def main():
    # --- (B) Make plt.show() a no-op to be extra safe
    _orig_show = plt.show
    plt.show = lambda *a, **k: None

    rows = []
    for y_name in Y_FILES:
        y_path = str(Y_DIR / y_name)
        thr_list = PER_FILE_THR.get(y_name, DEFAULT_THR_LIST)

        for thr in thr_list:
            print(f"\n=== Running y={y_name}, thr={thr} ===")

            try:
                real_r2, shuffled_scores, n_features = run_ridge_decoding(
                    x_path=X_PATH,
                    y_path=y_path,
                    n_shuffles=N_SHUFFLES,
                    train_frac=TRAIN_FRAC,
                    gap=GAP,
                    alpha=ALPHA,
                    rng=SEED,
                    use_filter=USE_FILTER,
                    corr_method=CORR_METHOD,
                    thr=thr,
                    plot=False,
                )

                shuf_mean, shuf_std, p_value, z_score = permutation_significance(real_r2, shuffled_scores)
                status = "ok"
                error_msg = ""

            except Exception as e:
                print(f"  -> Skipping: {e}")
                real_r2 = float("nan")
                shuf_mean = float("nan")
                shuf_std = float("nan")
                p_value = float("nan")
                z_score = float("nan")
                n_features = 0
                status = "skipped"
                error_msg = str(e)

            rows.append({
                "y_file": y_name,
                "thr": thr,
                "n_features": n_features,
                "real_r2": real_r2,
                "shuffled_mean": shuf_mean,
                "shuffled_std": shuf_std,
                "p_value": p_value,
                "z_score": z_score,
                "n_shuffles": N_SHUFFLES,
                "train_frac": TRAIN_FRAC,
                "gap": GAP,
                "alpha": ALPHA,
                "use_filter": int(USE_FILTER),
                "status": status,
                "error": error_msg[:200],
            })

            plt.close('all')

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "y_file", "thr", "n_features",
                "real_r2", "shuffled_mean", "shuffled_std",
                "p_value", "z_score",
                "n_shuffles", "train_frac", "gap", "alpha", "use_filter",
                "status", "error"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # restore show
    plt.show = _orig_show

    print(f"\nSaved summary to: {OUT_CSV.resolve()}")
    # Optional: pretty print results
    try:
        import pandas as pd
        df = pd.DataFrame(rows).sort_values(["y_file", "thr"])
        print(df.to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
