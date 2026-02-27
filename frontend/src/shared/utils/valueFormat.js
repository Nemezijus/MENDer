/**
 * Shared value formatting helpers.
 *
 * These are used across result renderers (ensembles + tuning analytics) to keep
 * formatting stable and avoid ad-hoc local helpers.
 */

export function safeNum(x) {
  if (x === null || x === undefined || x === '') return null;
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

export function fmtPct(x, digits = 1, placeholder = '—') {
  const n = safeNum(x);
  if (n == null) return placeholder;
  return `${(n * 100).toFixed(digits)}%`;
}

/**
 * Numeric formatter with a placeholder, preserving integers as integers.
 */
export function fmt(x, digits = 3, placeholder = '—') {
  const n = safeNum(x);
  if (n == null) return placeholder;
  return Number.isInteger(n) ? String(n) : n.toFixed(digits);
}

/**
 * Loose formatter used by some analytics panels.
 *
 * Matches legacy behavior:
 * - null/undefined/NaN -> String(x)
 * - number -> toFixed(digits)
 * - otherwise -> String(x)
 */
export function fmtAny(x, digits = 3) {
  if (x == null || Number.isNaN(x)) return String(x);
  if (typeof x === 'number') return x.toFixed(digits);
  return String(x);
}
