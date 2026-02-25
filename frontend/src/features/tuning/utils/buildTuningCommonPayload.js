import { compactPayload } from '../../../shared/utils/compactPayload.js';
import {
  buildDataPayload,
  buildEvalPayload,
  buildFeaturesPayload,
  buildScalePayload,
  buildSplitPayload,
} from '../../../shared/utils/payload/index.js';

function parseSeed({ shuffle, seed } = {}) {
  if (shuffle === false) return undefined;
  if (seed === '' || seed == null) return undefined;
  const n = parseInt(seed, 10);
  return Number.isFinite(n) ? n : undefined;
}

/**
 * Build the common payload skeleton for tuning endpoints.
 *
 * Convention:
 * - overrides-only (unset values omitted)
 * - split mode is k-fold for tuning panels
 * - seed is carried in eval (mirrors current tuning payloads)
 */
export function buildTuningCommonPayload({
  data,
  features,
  scaleMethod,
  model,
  split,
  evalMetric,
} = {}) {
  const parsedSeed = parseSeed({ shuffle: split?.shuffle, seed: split?.seed });

  return compactPayload({
    data: buildDataPayload(data),
    split: buildSplitPayload({
      mode: 'kfold',
      nSplits: split?.nSplits,
      stratified: split?.stratified,
      shuffle: split?.shuffle,
      // NOTE: do not pass seed here; tuning uses eval.seed.
    }),
    scale: buildScalePayload({ method: scaleMethod }),
    features: buildFeaturesPayload(features),
    model,
    eval: buildEvalPayload({
      metric: evalMetric,
      seed: parsedSeed,
    }),
  });
}
