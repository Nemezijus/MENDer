# utils/datafetch/allen_institute.py
from __future__ import annotations

import os
from typing import Union
import warnings

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
        "AllenSDK (>=2.16,<3) is required. Install with: pip install 'allensdk>=2.16,<3'\n"
        f"Import error: {e}"
    )


def get_vb_cache(cache_dir: str = "data/allen_cache") -> VisualBehaviorOphysProjectCache:
    """Create/return an S3-backed VisualBehavior cache under cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    return VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)


def load_visual_behavior_experiment(
    experiment_id: Union[int, str],
    *,
    cache_dir: str = "data/allen_cache",
):
    """
    Download (if needed) and return a Visual Behavior BehaviorOphysExperiment.

    This function **only fetches**; no preprocessing/feature-building here.
    """
    exp_id = int(experiment_id)
    cache = get_vb_cache(cache_dir)
    print(f"[INFO] Loading VisualBehavior ophys experiment {exp_id} …")
    exp = cache.get_behavior_ophys_experiment(exp_id)
    # Quick presence checks
    _ = exp.ophys_timestamps
    _ = exp.stimulus_presentations
    return exp


if __name__ == "__main__":
    # Minimal manual test: python -m utils.datafetch.allen_institute <ophys_experiment_id>
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m utils.datafetch.allen_institute <ophys_experiment_id>")
        raise SystemExit(1)
    exp = load_visual_behavior_experiment(sys.argv[1])
    print("[DONE] Loaded. stim columns:", list(exp.stimulus_presentations.columns))
