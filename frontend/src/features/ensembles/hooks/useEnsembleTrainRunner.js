import { useCallback, useState } from 'react';

import { useDataStore } from '../../dataFiles/state/useDataStore.js';
import { useResultsStore } from '../../results/state/useResultsStore.js';
import { useModelArtifactStore } from '../../modelArtifacts/state/useModelArtifactStore.js';

import { toErrorText } from '../../../shared/utils/errors.js';
import { runEnsembleTrainRequest } from '../api/ensemblesApi.js';

/**
 * Shared training runner for ensemble panels.
 *
 * This hook keeps request lifecycle (loading/error), validates that
 * inspected data exists, and standardizes success side-effects.
 */
export function useEnsembleTrainRunner() {
  const inspectReport = useDataStore((s) => s.inspectReport);

  const setTrainResult = useResultsStore((s) => s.setTrainResult);
  const setActiveResultKind = useResultsStore((s) => s.setActiveResultKind);
  const setArtifact = useModelArtifactStore((s) => s.setArtifact);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  const clearError = useCallback(() => setErr(null), []);

  const runTrain = useCallback(
    async ({ payload, buildPayload, noInspectMessage, clearPrevious = false } = {}) => {
      setErr(null);

      if (clearPrevious) {
        setTrainResult(null);
        setActiveResultKind(null);
        setArtifact(null);
      }

      if (!inspectReport || inspectReport?.n_samples <= 0) {
        setErr(
          noInspectMessage ||
            'No inspected training data. Please upload and inspect your data first.',
        );
        return null;
      }

      setLoading(true);
      try {
        const p = payload ?? (typeof buildPayload === 'function' ? buildPayload() : null);
        if (!p) throw new Error('Payload is empty.');

        const result = await runEnsembleTrainRequest(p);

        setTrainResult(result);
        setActiveResultKind('train');
        if (result?.artifact) setArtifact(result.artifact);

        setLoading(false);
        return result;
      } catch (e) {
        setLoading(false);
        setErr(toErrorText(e));
        return null;
      }
    },
    [inspectReport, setTrainResult, setActiveResultKind, setArtifact],
  );

  return { loading, err, setErr, clearError, runTrain };
}
