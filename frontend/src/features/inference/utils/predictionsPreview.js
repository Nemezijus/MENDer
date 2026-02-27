import { parseNumber, fmt3 } from '../../../shared/utils/numberFormat.js';

export { prettifyHeader, buildHeaderTooltip } from '../../../shared/utils/decoderHeaders.js';

export function pickPreviewColumns(rows) {
  if (!rows || rows.length === 0) return [];
  const keys = new Set();
  rows.forEach((r) => Object.keys(r || {}).forEach((k) => keys.add(k)));

  const preferred = [
    'index',
    'trial_id',
    'y_true',
    'y_pred',
    'correct',
    'residual',
    'abs_error',
    'margin',
    'decoder_score',
  ];

  // Match the decoder table / export ordering: scores first, then probabilities.
  const scoreCols = [...keys].filter((k) => k.startsWith('score_')).sort();
  const pCols = [...keys].filter((k) => k.startsWith('p_')).sort();
  const rest = [...keys]
    .filter((k) => !preferred.includes(k) && !k.startsWith('p_') && !k.startsWith('score_'))
    .sort();

  const out = [];
  preferred.forEach((k) => keys.has(k) && out.push(k));
  out.push(...scoreCols, ...pCols, ...rest);
  return out;
}

export function renderPreviewCell(col, value) {
  if (value === null || value === undefined) return '—';

  if (col === 'correct') {
    const isTrue = value === true || value === 'true';
    return isTrue ? 'true' : 'false';
  }

  const num = parseNumber(value);
  if (num !== null) return fmt3(num);

  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return String(value);
}
