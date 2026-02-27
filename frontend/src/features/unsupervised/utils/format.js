import { fmtNumber as sharedFmtNumber } from '../../../shared/utils/numberFormat.js';

export function fmtNumber(v, { digits = 3, empty = '—' } = {}) {
  // Preserve prior unsupervised-table behavior:
  // - null/undefined -> empty placeholder
  // - non-numeric non-null -> stringify
  return sharedFmtNumber(v, { digits, empty, passthroughNonNumber: true });
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
