// Decoder outputs helpers (formatting + split detection).

export { prettifyHeader, buildHeaderTooltip } from '../../../shared/utils/decoderHeaders.js';

export function getSummary(decoder) {
  if (!decoder) return null;
  return decoder.summary || decoder.decoder_summary || decoder.decoderSummary || decoder.decoderSummaries || null;
}

export function firstKey(obj, candidates) {
  if (!obj) return null;
  for (const k of candidates) {
    if (Object.prototype.hasOwnProperty.call(obj, k)) return k;
  }
  return null;
}

export function detectSplitType(trainResult) {
  const splitCandidates = [
    trainResult?.split,
    trainResult?.data_split,
    trainResult?.dataSplit,
    trainResult?.split_type,
    trainResult?.splitType,
    trainResult?.eval?.split,
    trainResult?.eval?.split_type,
    trainResult?.eval?.splitType,
    trainResult?.eval?.split_method,
    trainResult?.eval?.splitMethod,
    trainResult?.model_card?.split,
    trainResult?.modelCard?.split,
  ];

  const raw = splitCandidates.find((x) => typeof x === 'string' && x.trim().length > 0);
  const s = String(raw || '').toLowerCase();

  if (s.includes('kfold') || s.includes('k-fold') || s.includes('cv')) return 'kfold';
  if (s.includes('hold') || s.includes('test') || s.includes('split')) return 'holdout';

  return 'unknown';
}
