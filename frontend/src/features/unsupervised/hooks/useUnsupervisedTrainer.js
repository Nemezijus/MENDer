import { useCallback, useState } from 'react';
import { useShallow } from 'zustand/react/shallow';

import { useDataStore } from '../../dataFiles/state/useDataStore.js';
import { useResultsStore } from '../../results/state/useResultsStore.js';
import { useModelArtifactStore } from '../../modelArtifacts/state/useModelArtifactStore.js';
import { useSettingsStore } from '../../settings/state/useSettingsStore.js';
import { useFeatureStore } from '../../../shared/state/useFeatureStore.js';

import { toErrorText } from '../../../shared/utils/errors.js';
import {
  buildDataPayload,
  buildFeaturesPayload,
  buildScalePayload,
} from '../../../shared/utils/payload/index.js';

import { runUnsupervisedTrainRequest } from '../api/unsupervisedApi.js';
import { useUnsupervisedStore } from '../state/useUnsupervisedStore.js';
import { buildUnsupervisedEvalPayload } from '../utils/buildUnsupervisedEvalPayload.js';

/**
 * Unsupervised training request runner.
 *
 * Keeps UnsupervisedTrainingPanel mostly presentational.
 */
export function useUnsupervisedTrainer() {
  // Data selections come from the global Upload tab.
  const { xPath, yPath, npzPath, xKey, yKey } = useDataStore(
    useShallow((s) => ({
      xPath: s.xPath,
      yPath: s.yPath,
      npzPath: s.npzPath,
      xKey: s.xKey,
      yKey: s.yKey,
    })),
  );

  // Global Settings.
  const scaleMethod = useSettingsStore((s) => s.scaleMethod);

  const featureCtx = useFeatureStore(
    useShallow((s) => ({
      method: s.method,
      pca_n: s.pca_n,
      pca_var: s.pca_var,
      pca_whiten: s.pca_whiten,
      lda_n: s.lda_n,
      lda_solver: s.lda_solver,
      lda_shrinkage: s.lda_shrinkage,
      lda_tol: s.lda_tol,
      sfs_k: s.sfs_k,
      sfs_direction: s.sfs_direction,
      sfs_cv: s.sfs_cv,
      sfs_n_jobs: s.sfs_n_jobs,
    })),
  );

  // Unsupervised config.
  const { model, fitScope, metrics, includeClusterProbabilities, embeddingMethod } =
    useUnsupervisedStore(
      useShallow((s) => ({
        model: s.model,
        fitScope: s.fitScope,
        metrics: s.metrics,
        includeClusterProbabilities: s.includeClusterProbabilities,
        embeddingMethod: s.embeddingMethod,
      })),
    );

  // Outputs.
  const setTrainResult = useResultsStore((s) => s.setTrainResult);
  const setActiveResultKind = useResultsStore((s) => s.setActiveResultKind);
  const setArtifact = useModelArtifactStore((s) => s.setArtifact);

  // Local lifecycle state.
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const clearError = useCallback(() => setError(null), []);

  const runTraining = useCallback(async () => {
    const hasX = !!(xPath || npzPath);
    if (!hasX) {
      setError('Select a Feature matrix (X) in Upload data & models.');
      return null;
    }

    if (!model?.algo) {
      setError('Select an unsupervised algorithm first.');
      return null;
    }

    setError(null);
    setLoading(true);
    try {
      const payload = {
        task: 'unsupervised',
        data: buildDataPayload({
          xPath,
          yPath,
          npzPath,
          xKey: xKey?.trim() || undefined,
          yKey: yKey?.trim() || undefined,
        }),
        ...(fitScope !== undefined ? { fit_scope: fitScope } : {}),
        scale: buildScalePayload({ method: scaleMethod }),
        features: buildFeaturesPayload(featureCtx),
        model,
        eval: buildUnsupervisedEvalPayload({
          metrics,
          includeClusterProbabilities,
          embeddingMethod,
        }),
      };

      const resp = await runUnsupervisedTrainRequest(payload);

      setTrainResult(resp);
      setActiveResultKind('train');
      if (resp?.artifact) setArtifact(resp.artifact);

      setLoading(false);
      return resp;
    } catch (e) {
      setLoading(false);
      setError(toErrorText(e));
      return null;
    }
  }, [
    xPath,
    yPath,
    npzPath,
    xKey,
    yKey,
    fitScope,
    scaleMethod,
    featureCtx,
    model,
    metrics,
    includeClusterProbabilities,
    embeddingMethod,
    setTrainResult,
    setActiveResultKind,
    setArtifact,
  ]);

  return {
    loading,
    error,
    setError,
    clearError,
    runTraining,
  };
}
