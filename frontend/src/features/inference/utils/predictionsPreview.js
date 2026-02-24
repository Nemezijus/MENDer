function parseNumber(v) {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    const x = Number(v);
    if (Number.isFinite(x)) return x;
  }
  return null;
}

function fmt3(v) {
  if (typeof v === 'number' && Number.isFinite(v)) {
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(3);
  }
  return '—';
}

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

export function prettifyHeader(key) {
  if (!key) return '';
  if (key.startsWith('p_')) return `p=${key.slice(2)}`;
  if (key.startsWith('score_')) return `score=${key.slice(6)}`;
  return key.replaceAll('_', ' ');
}

export function buildHeaderTooltip(key) {
  if (key === 'index') return 'Row index within the preview.';
  if (key === 'trial_id') return 'Optional trial identifier (if provided).';
  if (key === 'y_true') return 'Ground-truth label/value for this sample (if provided).';
  if (key === 'y_pred') return 'Model-predicted label/value for this sample.';
  if (key === 'correct') return 'Whether prediction matches the ground truth.';
  if (key === 'residual') return 'Residual = (y_pred − y_true). Only present when y_true is provided (regression).';
  if (key === 'abs_error')
    return 'Absolute error = |y_pred − y_true|. Only present when y_true is provided (regression).';
  if (key === 'decoder_score')
    return 'Binary decision value (decision_function). More positive typically means stronger evidence for the positive class.';
  if (key === 'margin')
    return 'Confidence proxy: usually (top score − runner-up score). Larger margin = more confident.';
  if (key.startsWith('p_')) {
    const c = key.slice(2);
    return `Predicted probability for class ${c}. (Rows sum to ~1 across all p=... columns.)`;
  }
  if (key.startsWith('score_')) {
    const c = key.slice(6);
    return `Raw decision score (logit/decision value) for class ${c}. Higher score usually means higher probability.`;
  }
  return null;
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
