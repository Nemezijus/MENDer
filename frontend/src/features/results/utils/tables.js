// Small table + CSV helpers for results previews.

import {
  isEmptyCell,
  pickColumns as pickColumnsBase,
} from '../../../shared/utils/previewTable.js';

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

export { isEmptyCell };

/**
 * Pick stable, human-friendly columns for a preview table.
 * - keeps common preferred columns first
 * - keeps p_* and score_* together
 * - drops columns that are entirely empty in the preview
 */
export function pickColumns(rows) {
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

  return pickColumnsBase(rows, {
    preferred,
    groupPrefixes: ['p_', 'score_'],
    alwaysInclude: ['index'],
  });
}
