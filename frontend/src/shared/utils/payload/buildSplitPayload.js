import { compactPayload } from '../compactPayload.js';

/**
 * Build the "split" section for holdout or k-fold.
 *
 * Note:
 * - Mode is included as a discriminator.
 * - All other fields are override-only.
 */
export function buildSplitPayload({
  mode,
  trainFrac,
  nSplits,
  stratified,
  shuffle,
  seed,
} = {}) {
  if (!mode) return {};

  return compactPayload(
    mode === 'holdout'
      ? {
          mode: 'holdout',
          train_frac: trainFrac,
          stratified,
          shuffle,
          seed,
        }
      : {
          mode: 'kfold',
          n_splits: nSplits,
          stratified,
          shuffle,
          seed,
        },
  );
}
