// Shared helpers for model-parameter section components.
//
// Variant A goal: the store should hold ONLY explicit overrides.
// - UI may display engine defaults via placeholders / effectiveValue().
// - When the user sets a value equal to the engine default, we clear the override (set undefined).

function isPlainObject(x) {
  return x != null && typeof x === 'object' && !Array.isArray(x);
}

function deepEqual(a, b) {
  if (a === b) return true;

  // Handle NaN
  if (typeof a === 'number' && typeof b === 'number') {
    if (Number.isNaN(a) && Number.isNaN(b)) return true;
  }

  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i += 1) {
      if (!deepEqual(a[i], b[i])) return false;
    }
    return true;
  }

  if (isPlainObject(a) && isPlainObject(b)) {
    const ak = Object.keys(a);
    const bk = Object.keys(b);
    if (ak.length !== bk.length) return false;
    for (const k of ak) {
      if (!Object.prototype.hasOwnProperty.call(b, k)) return false;
      if (!deepEqual(a[k], b[k])) return false;
    }
    return true;
  }

  return false;
}

function formatDefaultValue(v) {
  if (v === null) return 'none';
  if (v === undefined) return '';

  if (typeof v === 'string') return v;
  if (typeof v === 'number') return String(v);
  if (typeof v === 'boolean') return v ? 'true' : 'false';

  if (Array.isArray(v)) {
    // Keep it readable for short arrays.
    return v.length <= 8 ? v.map((x) => String(x)).join(', ') : JSON.stringify(v);
  }

  if (isPlainObject(v)) {
    try {
      return JSON.stringify(v);
    } catch {
      return String(v);
    }
  }

  return String(v);
}

export function effectiveValue(override, defVal) {
  return override !== undefined ? override : defVal;
}

export function defaultPlaceholder(defVal) {
  if (defVal === undefined) return undefined;
  return `Default: ${formatDefaultValue(defVal)}`;
}

export function overrideOrUndef(next, defVal) {
  if (next === undefined) return undefined;
  if (defVal === undefined) return next;
  return deepEqual(next, defVal) ? undefined : next;
}
