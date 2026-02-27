// Shared number parsing/formatting helpers.

export function parseNumber(v) {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    const x = Number(v);
    if (Number.isFinite(x)) return x;
  }
  return null;
}

// Format a value to 3 decimals (or return an empty placeholder).
// NOTE: This is intentionally "strict": non-numeric values become the empty placeholder.
export function fmt3(v, empty = '—') {
  const num = parseNumber(v);
  if (num === null) return empty;
  if (Number.isInteger(num)) return String(num);
  return num.toFixed(3);
}

// General formatter.
// Supports both call styles:
//   fmtNumber(v, 4)
//   fmtNumber(v, { digits: 4, empty: '—', passthroughNonNumber: true })
export function fmtNumber(v, digitsOrOpts = 4) {
  const opts =
    typeof digitsOrOpts === 'number'
      ? { digits: digitsOrOpts }
      : digitsOrOpts && typeof digitsOrOpts === 'object'
        ? digitsOrOpts
        : {};

  const { digits = 4, empty = '—', passthroughNonNumber = false } = opts;

  if (v === null || v === undefined) return empty;

  const num = parseNumber(v);
  if (num === null) {
    return passthroughNonNumber ? String(v) : empty;
  }

  if (Number.isInteger(num)) return String(num);
  return num.toFixed(digits);
}
