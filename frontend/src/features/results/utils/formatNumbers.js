// Small formatting helpers for results panels.

import { parseNumber, fmt3 } from '../../../shared/utils/numberFormat.js';

export { parseNumber, fmt3 };

export function fmtMaybe3(v) {
  if (v === null || v === undefined) return '—';
  const num = parseNumber(v);
  if (num === null) return String(v);
  return fmt3(num);
}

export function fmtMaybePct(v) {
  const num = parseNumber(v);
  if (num === null) return '—';
  return `${(num * 100).toFixed(1)}%`;
}

export function rangeText(minV, maxV) {
  const a = parseNumber(minV);
  const b = parseNumber(maxV);
  if (a === null || b === null) return '—';
  return `${fmt3(a)} … ${fmt3(b)}`;
}

export function binCenters(edges) {
  if (!Array.isArray(edges) || edges.length < 2) return [];
  const out = [];
  for (let i = 0; i < edges.length - 1; i++) {
    const a = parseNumber(edges[i]);
    const b = parseNumber(edges[i + 1]);
    if (a === null || b === null) continue;
    out.push((a + b) / 2);
  }
  return out;
}

export function chunk(arr, n) {
  const xs = Array.isArray(arr) ? arr : [];
  const size = typeof n === 'number' && n > 0 ? n : 1;
  const out = [];
  for (let i = 0; i < xs.length; i += size) out.push(xs.slice(i, i + size));
  return out;
}
