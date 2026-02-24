import { useCallback } from 'react';

import { useProductionDataStore } from '../../dataFiles/state/useProductionDataStore.js';
import { useResultsStore } from '../../results/state/useResultsStore.js';
import { useModelArtifactStore } from '../../modelArtifacts/state/useModelArtifactStore.js';
import {
  applyModelToData,
  downloadBlob,
  exportPredictions,
  saveBlobInteractive,
} from '../../modelArtifacts/api/modelsApi.js';
import { buildDataPayload } from '../../../shared/utils/payload/buildDataPayload.js';
import { toErrorText } from '../../../shared/utils/errors.js';

export function useApplyModelRunner() {
  const artifact = useModelArtifactStore((s) => s.artifact);

  const xPath = useProductionDataStore((s) => s.xPath);
  const yPath = useProductionDataStore((s) => s.yPath);
  const npzPath = useProductionDataStore((s) => s.npzPath);
  const xKey = useProductionDataStore((s) => s.xKey);
  const yKey = useProductionDataStore((s) => s.yKey);

  const dataReady = useProductionDataStore((s) => Boolean(s.xPath || s.npzPath));

  const applyResult = useResultsStore((s) => s.applyResult);
  const setApplyResult = useResultsStore((s) => s.setApplyResult);

  const isRunning = useResultsStore((s) => s.productionIsRunning);
  const setIsRunning = useResultsStore((s) => s.setProductionIsRunning);

  const error = useResultsStore((s) => s.productionError);
  const setError = useResultsStore((s) => s.setProductionError);

  const hasModel = Boolean(artifact);
  const canRun = hasModel && dataReady && !isRunning;

  const getDataPayload = useCallback(() => {
    return buildDataPayload({
      xPath: xPath || undefined,
      yPath: yPath || undefined,
      npzPath: npzPath || undefined,
      xKey: xKey?.trim() || undefined,
      yKey: yKey?.trim() || undefined,
    });
  }, [xKey, xPath, yKey, yPath, npzPath]);

  const runPrediction = useCallback(async () => {
    if (!artifact) return;
    setIsRunning(true);
    setError(null);
    try {
      const resp = await applyModelToData({
        artifactUid: artifact.uid,
        artifactMeta: artifact,
        data: getDataPayload(),
      });
      setApplyResult(resp);
    } catch (e) {
      console.error(e);
      setApplyResult(null);
      setError(toErrorText(e) || 'Prediction failed');
    } finally {
      setIsRunning(false);
    }
  }, [artifact, getDataPayload, setApplyResult, setError, setIsRunning]);

  const exportPredictionsCsv = useCallback(async () => {
    if (!artifact) return;
    try {
      setError(null);
      const suggestedName = 'predictions.csv';
      const { blob, filename } = await exportPredictions({
        artifactUid: artifact.uid,
        artifactMeta: artifact,
        data: getDataPayload(),
        filename: suggestedName,
      });

      const supported = typeof window !== 'undefined' && 'showSaveFilePicker' in window;
      if (supported) {
        await saveBlobInteractive(blob, filename);
      } else {
        downloadBlob(blob, filename);
      }
    } catch (e) {
      console.error(e);
      setError(toErrorText(e) || 'Export failed');
    }
  }, [artifact, getDataPayload, setError]);

  return {
    artifact,
    hasModel,
    canRun,
    isRunning,
    error,
    applyResult,
    runPrediction,
    exportPredictionsCsv,
  };
}
