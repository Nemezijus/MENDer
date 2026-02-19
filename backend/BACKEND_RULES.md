# Backend rules

This directory contains MENDer’s **Backend API layer**.

The backend’s job is to:
- expose a stable **HTTP API** (FastAPI)
- perform **boundary work** (request validation, response shaping)
- perform **backend-owned IO concerns** (path normalization/allowlisting, upload store, progress registry)
- delegate **all modelling/business logic** to the Engine via **`engine/api.py`**

The backend must remain:
- **thin** (no ML orchestration)
- **consistent** (one error policy, predictable routes)
- **safe** (container path allowlisting, opaque 500s)

---

## Directory responsibilities

### `backend/app/`
FastAPI application package.

- `main.py`
  - creates the FastAPI app
  - mounts routers under `/api/v1`
  - defines **global exception handlers** (400 for client errors, opaque 500s)
  - configures CORS and request logging
- `startup.py`
  - startup hooks (e.g., ensure upload directories exist)

#### `backend/app/routers/`
HTTP endpoints only. Routers should:
- validate inputs via Pydantic models (backend `models/v1/*` and/or `engine.contracts.*`)
- build typed engine configs (`engine.contracts.*`)
- call backend services (or directly call `engine.api` if trivial)
- **not** contain business logic

Routes are organized by resource via router prefixes:
- `/files/*`, `/data/*`, `/pipeline/*`, `/train/*`, `/models/*`, `/predict/*`, `/tuning/*`, `/ensembles/*`, `/schema/*`, `/progress/*`, `/health/*`

#### `backend/app/services/`
Backend boundary services. Services should:
- call **only** `engine.api` (and never reach into engine internals)
- do lightweight orchestration that is **backend-owned** (e.g., wiring the progress registry)
- coerce engine results into JSON-friendly dicts via `services/common/result_coercion.py`
- avoid raising HTTP exceptions (raise plain exceptions; HTTP mapping is global in `main.py`)

#### `backend/app/adapters/`
Backend adapters for external systems.

##### `backend/app/adapters/io/`
All file/path handling and dataset-loading boundary logic.

Responsibilities:
- normalize/validate user paths (dev vs docker rules)
- allowlist container roots (DATA_ROOT + UPLOAD_DIR)
- build `engine.contracts.run_config.DataModel`
- delegate parsing to engine readers via `engine.api.load_from_data_model`

Key files:
- `environment.py`: environment detection and directory roots (`DATA_ROOT`, `UPLOAD_DIR`)
- `path_resolver.py`: path allowlisting + shorthand resolution (`data/...`, `uploads/...`)
- `data_config.py`: build `DataModel` from request parameters
- `loader.py`: `load_X_optional_y(...)`
- `errors.py`: `LoadError` (raised for user IO/config issues)

#### `backend/app/models/v1/`
Backend-owned Pydantic request/response models.

Rules:
- prefer `Field(default_factory=...)` for mutable defaults
- keep models cohesive by endpoint domain:
  - `train_models.py`, `tuning_models.py`, `ensemble_models.py`, `decoder_models.py`, `metrics_models.py`
  - `artifact_api_models.py` (save/load/apply/export schemas)
  - `files_models.py` (upload/list schemas)

#### `backend/app/progress/`
Process-lifetime progress tracking for long-running operations.

- `registry.py`: `PROGRESS` store keyed by `progress_id`
- `callback.py`: callback wrapper used by engine-facing runners

### `backend/utils/`
Backend-local utilities.

- `upload_hashing.py`: content-addressed file writes
- `upload_index.py`: stable filename/display-name index for uploads

### `backend/run.py`
Backend entrypoint for local execution.

---

## Public API & dependency rules

### Public backend surface
- HTTP API served by FastAPI (routers mounted under `/api/v1`)

### Import rules
- Backend may import from the Engine:
  - **`engine.api`** (invocation surface)
  - **`engine.contracts.*`** (schemas/configs)
- Backend must not import engine internals (components/factories/registries/etc.)
- Engine must **never** import backend.

---

## Error handling policy

- Router code should not wrap exceptions unless it is transforming them into a *more specific* backend exception.
- Global exception handlers live in `backend/app/main.py`.
- Expected client errors:
  - `LoadError` → 400
  - `ValueError` → 400
  - artifact validation/cache errors → 410/422
- Unexpected failures:
  - catch-all `Exception` → **opaque 500** with `{"detail": "Internal server error"}`
  - full traceback is logged via `logging.exception(...)`

---

## How to extend the backend (scenarios)

This section lists **backend files** typically affected. Many features also require Engine and Frontend changes; those are noted where relevant.

### 1) Add a new ensemble model and wire it into the frontend

**Typical backend impact (often minimal):**
- If the new ensemble is fully expressed via Engine contracts + schema bundle and uses the existing `/ensembles/train` endpoint, backend changes may be **zero**.

**Backend files you may need to touch when the API shape changes:**
- `backend/app/models/v1/ensemble_models.py`
  - add/extend typed `EnsembleReport` models if you want strict typing (otherwise the `Dict[str, Any]` fallback will still work)
- `backend/app/routers/ensembles.py`
  - only if you introduce a new endpoint (e.g., `/ensembles/tune`)
- `backend/app/services/ensemble_train_service.py`
  - only if you add new service calls beyond `engine.api.train_ensemble`

**Always check:**
- Frontend consumes schema from `/api/v1/schema/*` (generated in Engine). Adding a new ensemble kind usually requires Engine changes (contracts + registry + schema bundle) and Frontend UI updates.


### 2) Add support for a new data file type (input)

Backend work usually falls into two buckets: **upload/list support** and **path/loader support**.

**Backend files typically affected:**
- `backend/app/routers/files.py`
  - add extension to `ALLOWED_EXTS` (upload + listing filters)
- `backend/app/adapters/io/data_config.py`
  - if the Engine `DataModel` needs new fields to describe the new file type
- `backend/app/adapters/io/loader.py`
  - if loading requires additional parameters at the boundary (still delegates parsing to Engine)

**Usually also required:**
- Engine: extend `engine.api.load_from_data_model` / readers to parse the new file type
- Frontend: allow selecting the new file type and providing any new keys/params


### 3) Add new derived statistical outputs for a model (not ML scoring metrics)

These “derived results” should be computed in the Engine (typically in `engine/reporting/*`) and returned by Engine use-cases.

**Backend files typically affected (to expose them to the UI):**
- `backend/app/models/v1/train_models.py`
  - add fields to `TrainResponse` (and/or unsupervised response models)
- `backend/app/models/v1/metrics_models.py`
  - add structured Pydantic models for the new derived stats (recommended)
- `backend/app/models/v1/decoder_models.py`
  - if the new values relate to per-sample decoder outputs
- `backend/app/models/v1/ensemble_models.py`
  - if ensembles return the same new derived stats

**Backend services/routers are usually unchanged**, as they pass through engine results via `to_payload(...)`.

---

## Quick checklist for backend changes

Before committing backend changes, verify:
- no business logic was added outside `engine/api.py`
- routers are pure boundary layers (validation + config assembly + delegation)
- IO rules remain safe in docker (only allow DATA_ROOT + UPLOAD_DIR)
- response models include any new fields you expect the frontend to display
- error policy stays centralized and opaque for 500s
