import { getAlgoLabel } from '../../../shared/constants/algoLabels.js';
import { titleCase } from './resultsFormat.js';

/**
 * Convert an algo key into a UI label.
 *
 * If the shared label registry doesn't know the key, fall back to a simple
 * title-cased version so dropdowns still look reasonable.
 */
export function algoLabelWithFallback(key) {
  const k = String(key || '');
  const lbl = getAlgoLabel(k);
  return lbl === k ? titleCase(k) : lbl;
}
