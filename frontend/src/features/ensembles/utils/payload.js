import {
  buildDataPayload,
  buildEvalPayload,
  buildFeaturesPayload,
  buildScalePayload,
  buildSplitPayload,
} from '../../../shared/utils/payload/index.js';
import { compactPayload } from '../../../shared/utils/compactPayload.js';

/**
 * Build common payload sections for ensemble training.
 *
 * @param {object} args
 * @param {object} args.dataInputs
 * @param {object} args.splitInputs
 * @param {string|undefined} args.scaleMethod
 * @param {any} args.featureCtx
 * @param {object} args.evalInputs
 */
export function buildCommonEnsemblePayload({
  dataInputs,
  splitInputs,
  scaleMethod,
  featureCtx,
  evalInputs,
}) {
  const data = buildDataPayload(dataInputs);
  const split = buildSplitPayload(splitInputs);
  const scale = buildScalePayload({ method: scaleMethod });
  const features = buildFeaturesPayload(featureCtx);
  const evalCfg = buildEvalPayload(evalInputs);

  return { data, split, scale, features, eval: evalCfg };
}

/**
 * Combine common sections with the ensemble section.
 *
 * @param {object} args
 * @param {object} args.common
 * @param {object} args.ensemble
 */
export function buildEnsembleTrainPayload({ common, ensemble }) {
  return compactPayload({ ...(common || {}), ensemble });
}
