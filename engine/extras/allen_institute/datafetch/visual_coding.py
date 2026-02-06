from __future__ import annotations

"""Allen Visual Coding 2P experiment fetcher (AllenSDK).

This module is *optional* and depends on AllenSDK.
"""

import os
from typing import Union
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

try:
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
except Exception as e:
    raise ImportError(
        "AllenSDK with BrainObservatoryCache is required for Visual Coding 2P.\n"
        "Try: pip install 'allensdk>=2.13,<2.16'\n"
        f"Import error: {e}"
    )


def get_vc_cache(cache_dir: str = "data/allen_vc_cache") -> BrainObservatoryCache:
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "manifest.json")
    return BrainObservatoryCache(manifest_file=manifest_path)


def load_vc_experiment(experiment_id: Union[int, str]):
    """Download (if needed) and return a VC-2P BrainObservatoryNwbDataSet."""

    exp_id = int(experiment_id)
    boc = get_vc_cache()
    print(f"[INFO] Loading VC-2P ophys experiment {exp_id} …")
    ds = boc.get_ophys_experiment_data(exp_id)
    # Basic presence checks
    _ = ds.get_dff_traces()
    return ds
