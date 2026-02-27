// Shared helpers for decoder-style table headers and tooltips.

export function prettifyHeader(key) {
  if (!key) return '';
  if (key.startsWith('p_')) return `p=${key.slice(2)}`;
  if (key.startsWith('score_')) return `score=${key.slice(6)}`;

  const map = {
    index: 'Index',
    fold_id: 'Fold',
    trial_id: 'Trial',
    y_true: 'True value',
    y_pred: 'Predicted value',
    residual: 'Residual',
    abs_error: 'Absolute error',
    correct: 'Correct',
    margin: 'Margin',
    decoder_score: 'Decoder score',
  };

  if (Object.prototype.hasOwnProperty.call(map, key)) return map[key];

  return key
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function buildHeaderTooltip(key) {
  if (key === 'index') return 'Row index within the preview.';
  if (key === 'fold_id') return 'Fold index for this row in k-fold CV (1..K).';
  if (key === 'trial_id') return 'Optional trial identifier (if provided).';
  if (key === 'y_true') return 'Ground-truth value/label for this sample (if provided).';
  if (key === 'y_pred') return 'Model-predicted value/label for this sample.';
  if (key === 'residual') return 'Residual = predicted − true (regression).';
  if (key === 'abs_error') return '|predicted − true| (regression).';
  if (key === 'correct') return 'Whether prediction matches the ground truth.';
  if (key === 'decoder_score') {
    return 'Binary decision value (decision_function). More positive usually means stronger evidence for the positive class.';
  }
  if (key === 'margin') {
    return 'Confidence proxy: usually (top score − runner-up score) or (top probability − runner-up). Larger margin = more confident.';
  }
  if (key.startsWith('p_')) {
    const c = key.slice(2);
    return `Predicted probability for class ${c}. (Rows sum to ~1 across all p=... columns.)`;
  }
  if (key.startsWith('score_')) {
    const c = key.slice(6);
    return `Raw decision score for class ${c}. Higher score usually means higher probability.`;
  }
  return null;
}
