// Small table + CSV helpers for results previews.

function toCsvValue(v) {
  if (v === null || v === undefined) return '';
  const s = String(v);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

export function buildCsv(rows, columns) {
  const header = (columns || []).map(toCsvValue).join(',');
  const lines = (rows || []).map((r) =>
    (columns || []).map((c) => toCsvValue(r?.[c])).join(','),
  );
  return [header, ...lines].join('\n');
}

export function isEmptyCell(v) {
  if (v === null || v === undefined) return true;
  if (typeof v === 'string') return v.trim() === '';
  return false;
}

/**
 * Pick stable, human-friendly columns for a preview table.
 * - keeps common preferred columns first
 * - keeps p_* and score_* together
 * - drops columns that are entirely empty in the preview
 */
export function pickColumns(rows) {
  if (!rows || rows.length === 0) return [];

  const keys = new Set();
  rows.forEach((r) => Object.keys(r || {}).forEach((k) => keys.add(k)));

  const nonEmptyKeys = new Set(
    [...keys].filter((k) => k === 'index' || rows.some((r) => !isEmptyCell(r?.[k]))),
  );

  const preferred = [
    'index',
    'fold_id',
    'trial_id',
    'y_true',
    'y_pred',
    'residual',
    'abs_error',
    'correct',
    'margin',
    'decoder_score',
  ];

  const pCols = [...nonEmptyKeys].filter((k) => k.startsWith('p_')).sort();
  const scoreCols = [...nonEmptyKeys].filter((k) => k.startsWith('score_')).sort();
  const rest = [...nonEmptyKeys]
    .filter((k) => !preferred.includes(k) && !k.startsWith('p_') && !k.startsWith('score_'))
    .sort();

  const out = [];
  preferred.forEach((k) => nonEmptyKeys.has(k) && out.push(k));
  out.push(...pCols, ...scoreCols, ...rest);
  return out;
}
