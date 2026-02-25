function parseNumber(v) {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    const x = Number(v);
    if (Number.isFinite(x)) return x;
  }
  return null;
}

export function fmtNumber(v, { digits = 3, empty = '—' } = {}) {
  const num = parseNumber(v);
  if (num === null) return v == null ? empty : String(v);
  if (Number.isInteger(num)) return String(num);
  return num.toFixed(digits);
}

export function fmt3(v) {
  return fmtNumber(v, { digits: 3 });
}

export function fmtCell(v, { empty = '—' } = {}) {
  if (v === null || v === undefined) return empty;
  if (typeof v === 'boolean') return v ? 'true' : 'false';

  if (Array.isArray(v) || (typeof v === 'object' && v !== null)) {
    try {
      return JSON.stringify(v);
    } catch {
      return String(v);
    }
  }

  return fmt3(v);
}
