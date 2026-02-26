import { compactPayload } from '../../../shared/utils/compactPayload.js';
import {
  buildDataPayload,
  buildEvalPayload,
  buildFeaturesPayload,
  buildScalePayload,
} from '../../../shared/utils/payload/index.js';

function parseSeed({ shuffle, seed } = {}) {
  // Tuning passes RNG seed via eval.seed (mirrors current backend contract).
  // If the user explicitly disables shuffling, omit the seed override.
  if (shuffle === false) return undefined;
  if (seed === '' || seed == null) return undefined;
  const n = parseInt(seed, 10);
  return Number.isFinite(n) ? n : undefined;
}

/**
 * Build the common payload skeleton for tuning endpoints.
 *
 * Principles
 * ----------
 * - Overrides only: do not inject Engine defaults in the frontend.
 * - Do not hardcode split mode: tuning endpoints accept SplitCVModel and will
 *   apply defaults (including mode="kfold") when fields are omitted.
 * - Metric is optional: Engine chooses the correct default by task when unset.
 *
 * Notes
 * -----
 * - The tuning request models require `split`, `scale`, `features`, and `eval`
 *   objects, but these may be empty `{}` to trigger backend defaults.
 * - Seed is carried in `eval.seed` (consistent with existing tuning flows).
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

  // SplitCVModel expects snake_case. We intentionally omit `mode` so the backend
  // can apply its discriminator default ("kfold") without frontend forcing.
  const splitCvOverrides = compactPayload({
    n_splits: split?.nSplits,
    stratified: split?.stratified,
    shuffle: split?.shuffle,
  });

  return compactPayload({
    data: buildDataPayload(data),
    split: splitCvOverrides,
    scale: buildScalePayload({ method: scaleMethod }),
    features: buildFeaturesPayload(features),
    model,
    eval: buildEvalPayload({
      metric: evalMetric,
      seed: parsedSeed,
    }),
  });
}
