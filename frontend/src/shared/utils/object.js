/**
 * Minimal "plain object" check.
 *
 * We treat objects with a null prototype as plain as well (common for JSON payloads).
 */
export function isPlainObject(v) {
  return (
    v !== null &&
    typeof v === 'object' &&
    (v.constructor === Object || Object.getPrototypeOf(v) === null)
  );
}
