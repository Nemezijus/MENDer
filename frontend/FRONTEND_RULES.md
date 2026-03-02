# Frontend rules

This directory contains MENDer’s **Frontend UI layer**.

The frontend’s job is to:
- provide a coherent **UI workflow** (data → config → run → results → artifacts)
- render forms and results based on the **Engine/Backend schema contract**
- manage **UI state** (Zustand) and **server state** (TanStack Query)
- call the backend via a small, consistent HTTP layer

The frontend must remain:
- **contract-driven** (no hardcoded BL defaults)
- **thin** (no ML/business logic)
- **predictable** (stable state shapes, small shared utilities)
- **stylable** (no ad-hoc inline styling; CSS lives in the stylesheet structure)

---

## Directory responsibilities

### `frontend/src/app/`
App shell and navigation.

- layout primitives (`layout/*`)
- route/section navigation (`navigation/*`)
- guards (`guards/*`) that handle “you can’t proceed until…” UI logic

**Rule:** keep Mantine components and Mantine system style props here. Only extract **non-Mantine** styles.

### `frontend/src/features/`
Feature modules. Each feature owns its state, API calls, components, and feature-scoped styles.

Current features:
- `dataFiles/` – selecting/uploading datasets
- `training/` – single-model training
- `tuning/` – grid/random/curve tuning flows
- `ensembles/` – bagging/voting/adaboost/xgboost
- `inference/` – predictions preview / predict runs
- `results/` – training results, decoder outputs, visualizations
- `modelArtifacts/` – save/load/apply/export model artifacts
- `unsupervised/` – clustering/unsupervised runs and diagnostics
- `settings/` – UI/behavior settings

**Rule:** feature code must not reach into other features. Shared code goes into `src/shared/*`.

### `frontend/src/shared/`
Cross-feature UI infrastructure.

- `shared/api/` – axios client + standardized request helpers
- `shared/schema/` – schema/defaults fetching + helpers
- `shared/state/` – shared store factories/helpers
- `shared/ui/` – reusable UI pieces (config cards, tables, shared styles)
- `shared/constants/` – shared label maps/options (presentation-only)
- `shared/content/` – help text blocks and docs content
- `shared/utils/` – reusable helpers (formatters, compact payload, error formatting)

### Styling directories
The project uses a predictable stylesheet structure (created during Step 5 extraction):

- `src/index.css` – Tailwind import + minimal global rules
- `src/styles/` – app-wide (cross-feature) CSS
- `src/shared/ui/styles/` – shared UI look (cards/forms/tables/buttons/etc.)
- `src/shared/content/styles/` – help text styling
- `src/features/styles/` – feature-wide shared CSS
- `src/features/<feature>/styles/` – feature-scoped CSS

---

## Public surface & dependency rules

### The frontend may import from:
- `src/shared/*`
- its own feature folder (`src/features/<feature>/*`)
- Mantine/React Query/Zustand/Plotly and other normal UI libraries

### The frontend must not:
- import anything from `engine/*` or `backend/*`
- duplicate backend/engine business rules (defaults, allowed values, compatibility)

**All model/config defaults and allowed option sets come from the backend schema endpoints.**

---

## Contract-driven defaults (critical)

### Rule: no hardcoded defaults
UI must not encode “default C=1” or “default split fraction=0.75”.

Instead:
- Fetch schema defaults from the backend (`/api/v1/schema/defaults`)
- Display defaults in UI (“Default: …”)
- Store only **user overrides** in Zustand
- Send minimal payloads where omitted keys mean “use engine default”

### Schema access
- `src/shared/schema/SchemaDefaultsContext.jsx` provides `useSchemaDefaults()`
- Helpers include:
  - `getModelDefaults(algo)`
  - `getModelMeta(algo)`
  - `getCompatibleAlgos(task)`
  - `getEnsembleDefaults(kind)`

### Payload policy
- Payloads should be compact: omit empty slices and omit values equal to defaults.
- Use `src/shared/utils/compactPayload.js` (and related helpers) to keep request bodies small.

### Artifact hydration rule
When a run completes, the returned artifact often includes defaults resolved by the engine.

- Do **not** copy these resolved defaults into override state.
- Only hydrate override state from a **loaded artifact** (user action), not from a **trained artifact**.

---

## State management rules

### Split state: server vs UI
- **Server state** (schema, files constraints, etc.) lives in **TanStack Query**.
- **UI state** (form values / overrides / panel selections) lives in **Zustand**.

### Store helpers
Use `src/shared/state/storeFactories.js` for common patterns:
- resetters
- shallow-merge setters for nested slices
- array update/remove helpers

### Controllers vs presentation
Large panels should separate orchestration from JSX:
- “Controller” hook does config assembly, derived values, handlers
- JSX component is mostly layout

Example pattern:
- `SingleModelTrainingPanel.jsx` + `useSingleModelTrainingPanelController.js`

---

## HTTP/API calling rules

### Use the shared HTTP helpers
Prefer `src/shared/api/http.js` helpers:
- `getJson`, `postJson`, `putJson`, `delJson`
- `postFormData`
- `postBlob` (for export/save endpoints)

### Error shaping
- Convert backend errors into readable UI messages via `src/shared/utils/errors.js`.
- Do not show raw `[object Blob]` errors (handled by `postBlob`).

---

## Styling rules (Tailwind + Mantine)

### Mantine-first rule
Mantine is the component system. Keep Mantine components and Mantine system style props.

**Do not extract Mantine style props** (e.g., `mt/fw/c/pt/size/tt`) if doing so requires:
- Mantine CSS variables workarounds
- selector specificity hacks against Mantine generated classes

### Extract only what is safe
Allowed extractions:
- non-Mantine inline styles (`style={{...}}`) → semantic `className` + CSS
- shared, non-conflicting layout helpers (wrappers, spacing, sticky header positioning)

### CSS naming
- Use semantic names: `votingEstimatorRow`, `predPreviewStickyTh`, not `minW180MaxW360Flex`.
- Avoid encoding specific numeric values into class names.

### Tailwind v4 notes
- Tailwind is installed via Vite plugin (`@tailwindcss/vite`).
- CSS files that use `@apply` should include `@reference "tailwindcss";`.

---

## Shared utilities policy

If a helper exists in `src/shared/utils/`, use it rather than duplicating.

Examples of consolidated shared helpers:
- `numberFormat.js` (parse/format numbers)
- `decoderHeaders.js` (prettify header + tooltips)
- `previewTable.js` (table preview helpers)
- `valueFormat.js` (generic fmt helpers)
- `object.js` (`isPlainObject`)

---

## Performance & bundle size

- Large help content is lazy-loaded via `React.lazy()`.
- Do not import heavy help blocks at module scope for always-visible panels.
- Prefer “intro” components that remain lightweight.

---

## How to extend the frontend (scenarios)

### 1) Add a new model algorithm
Most work should be **schema-driven**.

Typical frontend impact:
- Training UI should list the new algo automatically via `getCompatibleAlgos(task)`.
- Add any model-specific help content (optional) in `src/shared/content/help/`.
- If the algo needs custom UI sections, add them under `features/training/components/modelSelection/`.

Always check:
- Backend/engine schema bundle includes the new algo defaults/meta.
- Payload remains overrides-only.

### 2) Add a new metric or evaluation output
- Prefer adding rendering support in `features/results/`.
- If the metric is reused broadly, centralize its labels/meta in `src/shared/constants/metrics.js`.

### 3) Add a new ensemble kind
- UI lists kinds from schema; avoid hardcoded kind lists.
- Put shared ensemble styles in `features/ensembles/styles/ensemblePanel.css` and kind-specific styles in the corresponding file (`voting.css`, `xgboost.css`, etc.).

---

## Quick checklist (frontend)

Before committing frontend changes, verify:
- No new hardcoded defaults were added (schema-driven only)
- Zustand stores contain only user overrides
- Requests are compact (empty slices removed)
- No business logic was introduced (frontend stays UI-only)
- Inline styles were not added (except truly dynamic data-driven cases)
- New CSS classes are semantic and live in the correct `styles/` folder
