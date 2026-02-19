# Engine (Business Layer) rules

This directory contains MENDer’s **Business Layer** (BL), referred to as the **engine**.
The engine must remain runnable **without** the backend or frontend (i.e., via scripts).

The goal of this refactor is to keep the engine:
- **Clean**: low coupling, high cohesion
- **Stable**: a small, intentional public API
- **Extensible**: easy to add new model families (e.g., neural nets) without editing large switch/if blocks
- **Deterministic**: RNG, splits, and evaluation should be reproducible

---

## Directory responsibilities

The engine is organized into the following subpackages:

- `engine/contracts/`
  - **Pydantic contracts**: configuration schemas and (later) result schemas.
  - Must be **dependency-light** and stable.

- `engine/core/`
  - Small, broadly reusable primitives (typing helpers, shape coercion, task kind, RNG helpers).
  - No heavy ML code here.

- `engine/runtime/`
  - **Mutable process-lifetime services/state** (e.g., caches).
  - Keep this disciplined: avoid pulling runtime state into low-level components.

- `engine/types/`
  - Shared type aliases / lightweight typing helpers used across engine modules.
  - Use this to prevent circular imports and keep signatures clean.

- `engine/components/`
  - Concrete compute components: data loaders, preprocessors, models, evaluators, predictors, ensemble builders.
  - Components implement clear interfaces and are testable in isolation.

- `engine/registries/`
  - Registration/mapping layer: config types → component implementations.
  - This is where we achieve “Open/Closed” extension: add a new file + register it.

- `engine/factories/`
  - Assembly helpers: given contracts/configs, build the correct components/pipelines.
  - Factories are internal wiring (used by use-cases); external callers should not import factories.

- `engine/io/`
  - Parsing and persistence (artifact store, readers, exporters).
  - Adapters only: no model logic.

- `engine/reporting/`
  - Derived outputs for humans/UI: decoder outputs, diagnostics, summaries, tables.
  - **No training** here; reporting consumes fit/eval artifacts.

- `engine/use_cases/`
  - **Application services / façade**: end-to-end workflows (train, predict, tune, ensembles, unsupervised).
  - This becomes the **primary public invocation surface** for backend and scripts.

  Note: use-cases are considered internal implementation. External layers should call the engine via
  `engine/api.py`.

- `engine/extras/`
  - Non-core, optional datasets and examples.
  - **Nothing in the core engine may import `extras`.**

---

## Public API policy

The intended public API of the engine is:
- **Invocation surface:** `engine/api.py`
- **Typed schemas/contracts:** `engine/contracts/*`

The backend (and any scripts outside engine) should:
- import **only** `engine.api` for calling engine functionality
- import **only** `engine.contracts.*` for schemas/configs/results

Everything else should be treated as internal implementation detail.

---

## Dependency rules (import direction)

To prevent a “Jenga tower”, imports must follow these rules:

### Allowed imports

- `engine/contracts` → stdlib + pydantic only
- `engine/core` → may import `engine/contracts`
- `engine/components` → may import `engine/core`, `engine/contracts`
- `engine/registries` → may import `engine/components`, `engine/core`, `engine/contracts`
- `engine/io` → may import `engine/core`, `engine/contracts`
- `engine/reporting` → may import `engine/components`, `engine/core`, `engine/contracts`
- `engine/use_cases` → may import everything above
- `engine/extras` → may import anything in engine, but **engine must not import extras**

### Forbidden imports

- **No imports from** `backend/` or `frontend/` anywhere in `engine/`.
- No FastAPI-specific types or response models inside the engine.
- No UI/state concepts inside the engine.

---

## Reference: engine map and extension guides

This section is intentionally practical: it’s a quick “where does this go?” map.

### Engine map (quick reference)

- **`api.py`**
  - The **single public entrypoint** for external layers (backend, scripts).
  - Re-exports stable functions (train/predict/tune/export/preview/etc.).

- **`contracts/`**
  - Pydantic/typed schemas: configs, choices, and result structures.
  - This is the shared “language” between backend and engine.

- **`use_cases/`**
  - End-to-end workflows (training, prediction, tuning, ensembles, exports, pipeline preview).
  - Orchestration lives here; use-cases may call factories/components/reporting.

- **`factories/`**
  - Assembly of pipelines and components based on contracts.
  - Use-cases call factories; external layers should not.

- **`components/`**
  - Concrete compute units (“bricks”): models, splitters, scalers, predictors, evaluators, exporters.
  - Components should avoid orchestration logic.

- **`registries/`**
  - Mappings from “algo name / config type” → implementation.
  - This is where new implementations get wired in (Open/Closed extension).

- **`reporting/`**
  - Derived / human-facing outputs: diagnostics, decoder outputs, summaries, tables.
  - No training orchestration here.

- **`io/`**
  - Engine-owned persistence/IO: readers, artifact formats, export helpers.
  - Adapters only (no modelling logic).

- **`core/`**
  - Foundational utilities (e.g., RNG helpers, shared stats). Safe to depend on broadly.

- **`runtime/`**
  - Mutable process-lifetime state/services (caches). Depend on this sparingly.

- **`types/`**
  - Shared type aliases and light typing helpers.

- **`extras/`**
  - Optional integrations/examples. Core engine code must not import extras.


### How to add a new model (4 scenarios)

The pattern is intentionally consistent: **add a contract → add a builder → register it**.
Think of `logreg` as the canonical supervised example.

#### 1) Add a new classifier: `newClassifier`

Typical impacted files (engine):

1. `engine/contracts/model_families/<some_family>.py`
   - Add `NewClassifierConfig` (Pydantic model) with `algo: Literal["newClassifier"]` and hyperparams.
2. `engine/contracts/model_families/registry.py`
   - Include `NewClassifierConfig` in the `ModelConfig` union.
3. `engine/contracts/model_families/__init__.py`
   - Re-export `NewClassifierConfig`.
4. `engine/components/models/builders/classification.py`
   - Implement `NewClassifierBuilder` (config → sklearn estimator).
5. `engine/components/models/builders/__init__.py`
   - Export `NewClassifierBuilder`.
6. `engine/registries/builtins/models.py`
   - Register the algo/config → builder mapping.

**Total typical engine files impacted:** 6


#### 2) Add a new regressor: `newRegressor`

Same pattern as classifier, but the builder goes into regression:

1. `engine/contracts/model_families/<some_family>.py` (add `NewRegressorConfig`)
2. `engine/contracts/model_families/registry.py` (add to `ModelConfig` union)
3. `engine/contracts/model_families/__init__.py` (re-export)
4. `engine/components/models/builders/regression.py` (add builder)
5. `engine/components/models/builders/__init__.py` (export)
6. `engine/registries/builtins/models.py` (register)

**Total typical engine files impacted:** 6


#### 3) Add a new ensemble method: `newEnsemble`

Ensembles touch more surface area because they usually include dedicated reporting/accumulation.

Typical impacted files (engine):

- Config + wiring:
  1. `engine/contracts/choices.py` (extend `EnsembleKind`)
  2. `engine/contracts/ensemble_configs.py` (add `NewEnsembleConfig` + union)
  3. `engine/registries/builtins/ensembles.py` (register kind)
  4. `engine/components/ensembles/strategies.py` (add strategy)
  5. `engine/components/ensembles/builders/<newensemble>.py` (new builder module)

- Reporting integration (to behave like existing ensemble tabs):
  6. `engine/use_cases/ensembles/types.py` (extend report state)
  7. `engine/use_cases/ensembles/reports.py` (init/finalize accumulator)
  8. `engine/use_cases/ensembles/folds.py` (update report per fold)
  9. `engine/reporting/ensembles/reports/<newensemble>.py` (new update function)
  10. `engine/reporting/ensembles/<newensemble>/...` (new accumulator module(s))

**Total typical engine files impacted:** ~12 (mix of modified + new)


#### 4) Add a new unsupervised model: `newUnsupervised`

Same pattern as supervised models, using the unsupervised builder:

1. `engine/contracts/model_families/unsupervised.py` (add config)
2. `engine/contracts/model_families/registry.py` (add to union)
3. `engine/contracts/model_families/__init__.py` (re-export)
4. `engine/components/models/builders/unsupervised.py` (add builder)
5. `engine/components/models/builders/__init__.py` (export)
6. `engine/registries/builtins/models.py` (register)

**Total typical engine files impacted:** 6


### How to add a new metric

“Metric” can mean either:
- a **scalar score** used for evaluation/selection (train, CV, tuning), or
- a **diagnostic payload** shown to users/exported (confusion/ROC-like structures), or
- an **unsupervised clustering metric**, or
- an **ensemble-specific derived metric**.

#### A) Add a new supervised scalar metric (classification/regression)

Impacted files:
1. `engine/contracts/choices.py`
   - Add the new name to `ClassificationMetricName` or `RegressionMetricName`.
2. `engine/components/evaluation/metrics/registry.py`
   - Implement the metric and register it in the metric maps.
   - If it needs probabilities/decision scores, add it to `PROBA_METRICS`.

Typical impact: **2 files**.


#### B) Add a new classification diagnostics payload (structured outputs)

Examples: PR curves, calibration curves, extra confusion-derived summaries.

Typical impacted files:
1. `engine/contracts/metrics_configs.py` (optional flags)
2. `engine/components/evaluation/metrics_computer.py` (compute + attach payload)
3. `engine/components/evaluation/types.py` (payload typing)
4. `engine/factories/metrics_factory.py` (thread config)
5. `engine/use_cases/supervised_training/aggregate.py` (carry/aggregate)
6. `engine/reporting/training/metrics_payloads.py` (JSON-safe normalization if needed)

Typical impact: **~5–6 files**.


#### C) Add a new regression diagnostics statistic (post-hoc payload)

Typical impacted files:
1. `engine/reporting/training/regression_payloads.py` (compute/attach)
2. `engine/use_cases/supervised_training/aggregate.py` (only if aggregation needs changes)

Typical impact: **1–2 files**.


#### D) Add a new unsupervised metric (clustering quality)

Minimal support:
1. `engine/contracts/choices.py` (add to `UnsupervisedMetricName`)
2. `engine/components/evaluation/unsupervised_scoring.py` (implement)

Full support including defaults + unsupervised tuning/search:
3. `engine/contracts/unsupervised_configs.py` (add to defaults)
4. `engine/components/tuning/runners/common.py`
5. `engine/components/tuning/unsupervised/common.py`
6. `engine/components/tuning/unsupervised/curves.py`
7. `engine/components/tuning/unsupervised/searchcv/param_space.py`

Typical impact: **2 files** (minimal) or **6–7 files** (full support).


#### E) Add a new ensemble-specific derived metric

Examples: diversity/disagreement measures, vote entropy, member correlation.

This typically lives in ensemble reporting and fold update wiring:
- `engine/reporting/ensembles/...`
- `engine/use_cases/ensembles/*`

Typical impact: **~3–6 files** depending on which ensemble(s) you support.

---

## Design rules

### Single Responsibility (S in SOLID)
- Prefer small modules with one purpose.
- Avoid “mega modules” that mix unrelated concerns (e.g., metric definitions + evaluation glue + UI formatting).

### Interfaces and contracts
- When a component has multiple implementations (e.g., scalers, splitters, evaluators), define an interface/protocol.
- Components should accept and return typed structures rather than ad-hoc dicts.

### Errors
- Avoid broad `except Exception:` unless you *re-raise* with context or intentionally degrade with a clear marker.
- Reporting can be best-effort, but must not hide core compute failures.

### Determinism
- Any randomness must come from an explicit RNG manager / seed passed down.
- No `np.random.*` in deep code without a seed source.


---

## Smoke test

Use `scripts/smoke_engine_imports.py` to quickly detect import drift during refactors:

```bash
python scripts/smoke_engine_imports.py
```

If this fails, fix imports before proceeding.
