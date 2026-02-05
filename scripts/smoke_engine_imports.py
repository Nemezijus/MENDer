"""Smoke test: verify the new `engine/` package is importable.

Run from the repository root:

    python scripts/smoke_engine_imports.py

This is intended to fail fast during refactors if imports drift/break.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Ensure repo root is on sys.path.

    This allows running the script from any working directory.
    """

    # This file is <repo_root>/scripts/smoke_engine_imports.py
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def main() -> int:
    _ensure_repo_root_on_syspath()

    modules = [
        "engine",
        "engine.contracts",
        "engine.contracts.choices",
        "engine.contracts.decoder_configs",
        "engine.contracts.ensemble_configs",
        "engine.contracts.ensemble_run_config",
        "engine.contracts.eval_configs",
        "engine.contracts.feature_configs",
        "engine.contracts.metrics_configs",
        "engine.contracts.model_configs",
        "engine.contracts.run_config",
        "engine.contracts.scale_configs",
        "engine.contracts.split_configs",
        "engine.contracts.tuning_configs",
        "engine.contracts.types",
        "engine.contracts.unsupervised_configs",
        "engine.contracts.results",
        "engine.contracts.results.decoder",
        "engine.contracts.results.training",
        "engine.contracts.results.ensemble",
        "engine.contracts.results.prediction",
        "engine.contracts.results.tuning",
        "engine.contracts.results.unsupervised",
        "engine.core",
        "engine.components",
        "engine.registries",
        "engine.io",
        "engine.reporting",
        "engine.use_cases",
        "engine.compat",
        "engine.extras",
    ]

    failures: list[tuple[str, BaseException]] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except BaseException as e:  # noqa: BLE001 - this is a smoke test
            failures.append((mod, e))

    if failures:
        print("ENGINE IMPORT SMOKE TEST: FAILED\n")
        for mod, e in failures:
            print(f"- {mod}: {type(e).__name__}: {e}")
        print("\nFix imports before continuing the refactor.")
        return 1

    
    import engine.components.prediction.predicting
    import engine.components.prediction.decoder_extraction
    import engine.components.prediction.decoder_api
    import engine.reporting.prediction.prediction_results
    import engine.reporting.decoder.decoder_outputs
    import engine.components.evaluation.scoring

    print("ENGINE IMPORT SMOKE TEST: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
