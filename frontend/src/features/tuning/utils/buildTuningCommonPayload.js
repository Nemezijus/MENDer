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

function getKfoldDefaults(schemaDefaults) {
  return schemaDefaults?.split?.kfold?.defaults ?? null;
}

/**
 * Build the common payload skeleton for tuning endpoints.
 *
 * Variant A (overrides-only) still applies at the store level, but for tuning we
 * intentionally send *effective* defaults for split/scale/features so the backend
 * receives the same configuration the UI is displaying.
 *
 * Sources of defaults:
 * - split/scale/features defaults come from /schema/defaults (engine-owned)
 * - user overrides come from stores (may be undefined)
 *
 * Convention:
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
  schemaDefaults,
} = {}) {
  const kfoldDefaults = getKfoldDefaults(schemaDefaults);

  const effectiveNSplits = split?.nSplits ?? kfoldDefaults?.n_splits ?? undefined;
  const effectiveStratified =
    split?.stratified ?? kfoldDefaults?.stratified ?? undefined;
  const effectiveShuffle = split?.shuffle ?? kfoldDefaults?.shuffle ?? undefined;

  const parsedSeed = parseSeed({ shuffle: effectiveShuffle, seed: split?.seed });

  const scaleDefaultMethod = schemaDefaults?.scale?.defaults?.method ?? undefined;
  const effectiveScaleMethod = scaleMethod ?? scaleDefaultMethod;

  const featureDefaultMethod =
    schemaDefaults?.features?.defaults?.method ?? undefined;

  const hasEffectiveMethod =
    (features?.method ?? featureDefaultMethod) != null;

  const effectiveFeatures = hasEffectiveMethod
    ? { ...(features ?? {}), method: features?.method ?? featureDefaultMethod }
    : features;

  return compactPayload({
    data: buildDataPayload(data),
    split: buildSplitPayload({
      mode: 'kfold',
      nSplits: effectiveNSplits,
      stratified: effectiveStratified,
      shuffle: effectiveShuffle,
      // NOTE: do not pass seed here; tuning uses eval.seed.
    }),
    scale: buildScalePayload({ method: effectiveScaleMethod }),
    features: buildFeaturesPayload(effectiveFeatures),
    model,
    eval: buildEvalPayload({
      metric: evalMetric,
      seed: parsedSeed,
    }),
  });
}
