import { compactPayload } from '../compactPayload.js';

/**
 * Build the "eval" section.
 *
 * Convention:
 * - Overrides only.
 * - Decoder selection is typically left to engine defaults.
 */
export function buildEvalPayload({ metric, seed, nShuffles, progressId } = {}) {
  return compactPayload({
    metric,
    seed,
    n_shuffles: nShuffles,
    progress_id: progressId,
  });
}
