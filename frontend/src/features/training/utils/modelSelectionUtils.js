import { enumFromSubSchema, toSelectData } from '../../../shared/utils/schema/jsonSchema.js';

export function parseCsvFloats(s) {
  if (s == null) return null;
  const txt = String(s).trim();
  if (!txt) return null;
  const parts = txt
    .split(',')
    .map((x) => x.trim())
    .filter((x) => x.length);
  if (!parts.length) return null;
  const vals = parts.map((x) => Number(x)).filter((x) => Number.isFinite(x));
  return vals.length ? vals : null;
}

export function formatCsvFloats(arr) {
  if (!Array.isArray(arr) || !arr.length) return '';
  return arr.map((x) => String(x)).join(', ');
}

export function maxFeatToModeVal(v) {
  if (v == null) return { mode: 'none', value: null };
  if (v === 'sqrt' || v === 'log2') return { mode: v, value: null };
  if (typeof v === 'number' && Number.isFinite(v)) {
    return { mode: Number.isInteger(v) ? 'int' : 'float', value: v };
  }
  return { mode: 'none', value: null };
}

export function modeValToMaxFeat(mode, value) {
  if (mode === 'none') return null;
  if (mode === 'sqrt' || mode === 'log2') return mode;
  if (mode === 'int' || mode === 'float') {
    if (value == null || value === '') return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

export function makeSelectData(subSchema, key, fallbackEnum, opts) {
  return toSelectData(enumFromSubSchema(subSchema, key, fallbackEnum), opts);
}
