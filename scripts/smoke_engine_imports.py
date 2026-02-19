"""Smoke test: verify the new `engine/` package is importable.

Run from the repository root:

    python scripts/smoke_engine_imports.py

This is intended to fail fast during refactors if imports drift/break.
"""

from __future__ import annotations

import importlib
import subprocess
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
    repo_root = _ensure_repo_root_on_syspath()

    # Boundary guard: backend must only import Engine via engine.api + engine.contracts.
    chk = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "check_engine_boundary.py")],
        cwd=str(repo_root),
    )
    if chk.returncode != 0:
        return chk.returncode

    modules = [
        "engine",
        "engine.api",
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
        "engine.core.shapes",
        "engine.components",
        "engine.registries",
        "engine.registries.base",
        "engine.registries.models",
        "engine.registries.features",
        "engine.registries.splitters",
        "engine.registries.ensembles",
        "engine.registries.exporters",
        "engine.registries.metrics",
        "engine.registries.builtins",
        "engine.registries.builtins.models",
        "engine.registries.builtins.features",
        "engine.registries.builtins.splitters",
        "engine.registries.builtins.ensembles",
        "engine.registries.builtins.exporters",
        "engine.io",
        "engine.io.artifacts",
        "engine.io.readers",
        "engine.io.readers.auto_reader",
        "engine.io.readers.mat_reader",
        "engine.io.readers.tabular_reader",
        "engine.io.artifacts.store",
        "engine.io.artifacts.filesystem_store",
        "engine.io.artifacts.serialization",
        "engine.io.artifacts.meta",
        "engine.runtime",
        "engine.runtime.caches",
        "engine.runtime.caches.artifact_cache",
        "engine.runtime.caches.eval_outputs_cache",
        "engine.reporting",
        "engine.reporting.common",
        "engine.reporting.common.json_safety",
        "engine.reporting.training",
        "engine.reporting.training.metrics_payloads",
        "engine.reporting.training.decoder_payloads",
        "engine.reporting.training.regression_payloads",
        "engine.use_cases",
        "engine.use_cases._deps",
        "engine.use_cases.facade",
        "engine.use_cases.supervised_training",
        "engine.use_cases.unsupervised_training",
        "engine.use_cases.ensembles",
        "engine.use_cases.prediction",
        "engine.use_cases.tuning",
        "engine.reporting.ensembles.helpers",
        "engine.reporting.ensembles.reports",
        "engine.reporting.ensembles.reports.voting",
        "engine.reporting.ensembles.reports.bagging",
        "engine.reporting.ensembles.reports.adaboost",
        "engine.reporting.ensembles.reports.xgboost",

        # Segment 7: split mega-modules
        "engine.components.evaluation.metrics",
        "engine.reporting.diagnostics.clustering",
        "engine.contracts.model_families",
    ]

    failures: list[tuple[str, BaseException]] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except BaseException as e:  # noqa: BLE001 - this is a smoke test
            failures.append((mod, e))

    # Guard: core Engine imports should not pull in optional extras
    extras_loaded = [m for m in sys.modules if m == 'engine.extras' or m.startswith('engine.extras.') ]
    if extras_loaded:
        print('ENGINE IMPORT SMOKE TEST: FAILED\n')
        print('Core imports unexpectedly loaded engine.extras:')
        for m in sorted(set(extras_loaded)):
            print(f'- {m}')
        print('\nCore must not import extras; move dataset/example code under engine/extras and keep it optional.')
        return 1

    if failures:
        print("ENGINE IMPORT SMOKE TEST: FAILED\n")
        for mod, e in failures:
            print(f"- {mod}: {type(e).__name__}: {e}")
        print("\nFix imports before continuing the refactor.")
        return 1

    # Guardrail: core Engine packages must not import extras automatically.
    assert not any(m == 'engine.extras' or m.startswith('engine.extras.') for m in sys.modules), (
        'Engine core imported engine.extras unexpectedly; keep extras isolated.'
    )

    # Import extras package itself (should be lightweight).
    import engine.extras  # noqa: F401
    # A few extra explicit imports to exercise key submodules.
    import engine.components.prediction.predicting  # noqa: F401
    import engine.components.prediction.decoder_extraction  # noqa: F401
    import engine.components.prediction.decoder.api  # noqa: F401
    import engine.reporting.prediction.prediction_results  # noqa: F401
    import engine.reporting.decoder.decoder_outputs  # noqa: F401
    import engine.components.evaluation.scoring  # noqa: F401
    import engine.io.artifacts  # noqa: F401
    import engine.runtime.caches  # noqa: F401
    import engine.use_cases.artifacts  # noqa: F401
    import engine.use_cases.facade  # noqa: F401
    import engine.api  # noqa: F401
    import engine.use_cases.supervised_training  # noqa: F401
    import engine.use_cases.unsupervised_training  # noqa: F401
    import engine.use_cases.ensembles  # noqa: F401
    import engine.use_cases.prediction  # noqa: F401
    import engine.use_cases.tuning  # noqa: F401
    import engine.reporting.common.json_safety  # noqa: F401
    import engine.reporting.training.metrics_payloads  # noqa: F401

    # Exercise registries so built-ins must actually register.
    from engine.registries.models import list_model_algos
    from engine.registries.features import list_feature_methods
    from engine.registries.splitters import list_split_modes
    from engine.registries.ensembles import list_ensemble_kinds
    from engine.registries.exporters import list_export_formats

    assert list_model_algos(), "model registry looks empty"
    assert list_feature_methods(), "feature registry looks empty"
    assert list_split_modes(), "splitter registry looks empty"
    assert list_ensemble_kinds(), "ensemble registry looks empty"
    assert list_export_formats(), "exporter registry looks empty"

    print("ENGINE IMPORT SMOKE TEST: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
