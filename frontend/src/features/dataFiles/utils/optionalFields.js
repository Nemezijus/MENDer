/**
 * Variant A ("overrides-only") helpers:
 * - return undefined when a value is effectively unset (null/undefined/empty string)
 * - allows payload builders + compactPayload() to omit those keys entirely
 */

export function optString(v) {
  if (v == null) return undefined;
  const s = String(v).trim();
  return s ? s : undefined;
}

export function optNumber(v) {
  if (v == null) return undefined;
  return v;
}
