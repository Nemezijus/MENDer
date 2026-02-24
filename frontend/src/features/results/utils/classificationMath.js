import { safeMean, safeWeightedMean } from './stats.js';

export { safeMean, safeWeightedMean };

export function computeImbalanceInfo(supports) {
  const xs = Array.isArray(supports)
    ? supports.map((s) => (typeof s === 'number' && Number.isFinite(s) ? s : 0))
    : [];

  const nonZero = xs.filter((s) => s > 0);
  const ratio =
    nonZero.length >= 2 ? Math.max(...nonZero) / Math.min(...nonZero) : null;

  let desc = null;
  if (typeof ratio === 'number' && Number.isFinite(ratio)) {
    if (ratio <= 1.5) {
      desc = 'Classes are well balanced.';
    } else if (ratio <= 3) {
      desc = 'Classes are mildly imbalanced.';
    } else if (ratio <= 5) {
      desc = 'Classes are moderately imbalanced.';
    } else {
      desc = 'Classes are strongly imbalanced (majority class dominates the minority).';
    }
  }

  return { ratio, desc };
}
