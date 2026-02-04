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

- `engine/components/`
  - Concrete compute components: data loaders, preprocessors, models, evaluators, predictors, ensemble builders.
  - Components implement clear interfaces and are testable in isolation.

- `engine/registries/`
  - Registration/mapping layer: config types → component implementations.
  - This is where we achieve “Open/Closed” extension: add a new file + register it.

- `engine/io/`
  - Parsing and persistence (artifact store, readers, exporters).
  - Adapters only: no model logic.

- `engine/reporting/`
  - Derived outputs for humans/UI: decoder outputs, diagnostics, summaries, tables.
  - **No training** here; reporting consumes fit/eval artifacts.

- `engine/use_cases/`
  - **Application services / façade**: end-to-end workflows (train, predict, tune, ensembles, unsupervised).
  - This becomes the **primary public invocation surface** for backend and scripts.

- `engine/compat/`
  - Temporary shims / re-exports used during the refactor.
  - Must not become the long-term API.

- `engine/extras/`
  - Non-core, optional datasets and examples.
  - **Nothing in the core engine may import `extras`.**

---

## Public API policy

The intended public API of the engine is:
- `engine/use_cases/*` (and later `engine/api.py`)

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
- Avoid importing `engine/compat` from new code (compat is transitional).

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

## Transitional notes (Segment 1)

During early refactor segments, `engine/contracts/*` may re-export the existing `shared_schemas/*`.
This is intentional to allow switching imports to `engine.contracts` early, before moving files.

---

## Smoke test

Use `scripts/smoke_engine_imports.py` to quickly detect import drift during refactors:

```bash
python scripts/smoke_engine_imports.py
```

If this fails, fix imports before proceeding.
