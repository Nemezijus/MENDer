/**
 * Deeply removes keys with `undefined` values (and optionally empty strings / nulls)
 * from an object intended to be JSON-stringified.
 *
 * Useful for "overrides-only" payloads, where defaults are owned by the
 * engine/backend and the frontend should omit unset fields.
 */

function isPlainObject(v) {
  return (
    v !== null &&
    typeof v === 'object' &&
    (v.constructor === Object || Object.getPrototypeOf(v) === null)
  );
}

/**
 * @param {any} value
 * @param {object} [opts]
 * @param {boolean} [opts.dropEmptyString=true] - drop keys where value === ''
 * @param {boolean} [opts.dropNull=false] - drop keys where value === null
 */
export function compactPayload(value, opts = {}) {
  const { dropEmptyString = true, dropNull = false } = opts;

  if (Array.isArray(value)) {
    return value
      .map((v) => compactPayload(v, opts))
      .filter((v) => v !== undefined);
  }

  if (isPlainObject(value)) {
    const out = {};
    for (const [k, v] of Object.entries(value)) {
      if (v === undefined) continue;
      if (dropNull && v === null) continue;
      if (dropEmptyString && v === '') continue;

      const vv = compactPayload(v, opts);
      if (vv === undefined) continue;
      if (dropNull && vv === null) continue;
      if (dropEmptyString && vv === '') continue;

      out[k] = vv;
    }
    return out;
  }

  return value;
}
