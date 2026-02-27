/**
 * Small coercion helpers for override-only payload building.
 *
 * Mantine inputs may provide numbers or empty strings; stores often keep
 * either numbers or `undefined`.
 */

export function numOrUndef(v) {
  if (v === '' || v == null) return undefined;
  if (typeof v === 'number') return Number.isFinite(v) ? v : undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

export function intOrUndef(v) {
  const n = numOrUndef(v);
  if (n === undefined) return undefined;
  return Math.trunc(n);
}

export function boolOrUndef(v) {
  if (v === '' || v == null) return undefined;
  if (typeof v === 'boolean') return v;
  if (v === 'true') return true;
  if (v === 'false') return false;
  return undefined;
}
