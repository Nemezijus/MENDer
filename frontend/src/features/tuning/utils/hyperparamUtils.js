/**
 * Hyperparameter selection helpers.
 *
 * These are intentionally frontend-only utilities. They derive UI affordances
 * from the backend JSON schema for a selected model/algo.
 */

export function getParamSchema(sub, key) {
  if (!sub || !key) return null;
  return sub.properties?.[key] ?? null;
}

export function collectTypes(p) {
  const out = new Set();
  if (!p) return out;

  const direct = Array.isArray(p.type) ? p.type : p.type ? [p.type] : [];
  for (const t of direct) out.add(t);

  const unions = [...(p.anyOf ?? []), ...(p.oneOf ?? [])];
  for (const u of unions) {
    const ts = Array.isArray(u.type) ? u.type : u.type ? [u.type] : [];
    for (const t of ts) out.add(t);
  }

  return out;
}

export function collectEnumValues(p) {
  if (!p) return null;
  if (Array.isArray(p.enum)) return p.enum;

  // Pydantic often represents Optional[...] as anyOf: [{type: <real>}, {type: 'null'}].
  // We only want to treat a parameter as an enum if there are *actual* concrete values.
  // Otherwise Optional[int] could become a bogus enum of [null] in the UI.
  const unions = [...(p.anyOf ?? []), ...(p.oneOf ?? [])];
  const list = [];
  for (const x of unions) {
    if (Array.isArray(x.enum)) list.push(...x.enum);
    else if (x.const != null) list.push(x.const);
    else if (x.type === 'null') list.push(null);
  }

  // De-duplicate while preserving order (Object.is handles null correctly)
  const uniq = [];
  for (const v of list) {
    if (!uniq.some((u) => Object.is(u, v))) uniq.push(v);
  }

  // Ignore “enum lists” that are only [null]
  if (uniq.length === 1 && uniq[0] === null) return null;

  return uniq.length ? uniq : null;
}

export function humanizeParamName(key) {
  if (!key) return '';

  const map = {
    // common / shared
    n_estimators: 'Number of estimators',
    max_iter: 'Max iterations',
    n_jobs: 'N jobs',
    class_weight: 'Class weight',
    random_state: 'Random state',
    warm_start: 'Warm start',
    tol: 'Tolerance',

    // trees / forests
    max_depth: 'Max depth',
    min_samples_split: 'Min samples split',
    min_samples_leaf: 'Min samples leaf',
    min_impurity_decrease: 'Min impurity decrease',
    max_leaf_nodes: 'Max leaf nodes',
    min_weight_fraction_leaf: 'Min weight fraction leaf',
    max_features: 'Max features',
    max_samples: 'Max samples',
    oob_score: 'OOB score',
    ccp_alpha: 'CCP alpha',

    // linear models
    fit_intercept: 'Fit intercept',
    l1_ratio: 'L1 ratio',
    learning_rate: 'Learning rate schedule',
    eta0: 'Initial learning rate (eta0)',
    power_t: 'Power t',

    // HGB
    max_bins: 'Max bins',
    validation_fraction: 'Validation fraction',
    n_iter_no_change: 'N iter no change',
    l2_regularization: 'L2 regularization',
  };

  if (map[key]) return map[key];

  // Generic fallback: snake_case -> Title case
  return key.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase());
}

export function getParamInfo(sub, key) {
  const p = getParamSchema(sub, key);
  if (!p) return { kind: 'other', allowedValues: null };

  const enums = collectEnumValues(p);
  const types = collectTypes(p);

  // Boolean by explicit type or enum {true,false}
  if (types.has('boolean')) {
    return { kind: 'boolean', allowedValues: [true, false] };
  }

  if (enums) {
    const uniq = Array.from(new Set(enums));
    if (uniq.length === 2 && uniq.includes(true) && uniq.includes(false)) {
      return { kind: 'boolean', allowedValues: [true, false] };
    }
    return { kind: 'enum', allowedValues: enums };
  }

  if (types.has('number') || types.has('integer')) {
    return { kind: 'numeric', allowedValues: null };
  }

  return { kind: 'other', allowedValues: null };
}

export function parseScalar(raw) {
  const lower = String(raw).toLowerCase();

  if (lower === 'true' || lower === 'yes') return true;
  if (lower === 'false' || lower === 'no') return false;
  if (lower === 'none' || lower === 'null') return null;

  if (/^-?\d+$/.test(raw)) {
    const v = parseInt(raw, 10);
    return Number.isNaN(v) ? raw : v;
  }

  if (/^-?\d*\.\d+$/.test(raw)) {
    const v = parseFloat(raw);
    return Number.isNaN(v) ? raw : v;
  }

  return raw;
}

export function countDecimals(str) {
  const m = String(str).match(/\.(\d+)/);
  return m ? m[1].length : 0;
}

export function formatValueForDisplay(v, displayPrecision) {
  if (typeof v === 'number' && Number.isFinite(v) && displayPrecision != null) {
    return v.toFixed(displayPrecision);
  }
  return String(v);
}

export function summarizeValues(values, maxDisplay = 10) {
  const arr = Array.isArray(values) ? values : [];
  if (arr.length <= maxDisplay) return arr;
  return [...arr.slice(0, maxDisplay - 1), '…', arr[arr.length - 1]];
}
