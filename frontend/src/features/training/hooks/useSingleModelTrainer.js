import { useCallback, useEffect, useRef, useState } from 'react';

import { useDataStore } from '../../dataFiles/state/useDataStore.js';
import { useResultsStore } from '../../results/state/useResultsStore.js';
import { useModelArtifactStore } from '../../modelArtifacts/state/useModelArtifactStore.js';
import { useSchemaDefaults } from '../../../shared/schema/SchemaDefaultsContext.jsx';
import { useFeatureStore } from '../../../shared/state/useFeatureStore.js';
import { useSettingsStore } from '../../settings/state/useSettingsStore.js';
import { useModelConfigStore } from '../state/useModelConfigStore.js';
import { useTrainingStore } from '../state/useTrainingStore.js';

import { runTrainRequest } from '../api/trainApi.js';
import { fetchProgress } from '../api/progressApi.js';

import { toErrorText } from '../../../shared/utils/errors.js';
import { getDefaultSplitMode } from '../../../shared/utils/splitMode.js';
import {
  buildDataPayload,
  buildEvalPayload,
  buildFeaturesPayload,
  buildScalePayload,
  buildSplitPayload,
} from '../../../shared/utils/payload/index.js';

function makeProgressId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Orchestrates the train request + optional progress polling.
 * Keeps SingleModelTrainingPanel mostly presentational.
 */
export function useSingleModelTrainer() {
  // --- data inputs --------------------------------------------------------
  const xPath = useDataStore((s) => s.xPath);
  const yPath = useDataStore((s) => s.yPath);
  const npzPath = useDataStore((s) => s.npzPath);
  const xKey = useDataStore((s) => s.xKey);
  const yKey = useDataStore((s) => s.yKey);
  const inspectReport = useDataStore((s) => s.inspectReport);
  const dataReady = !!inspectReport && inspectReport?.n_samples > 0;

  // --- schema defaults (for effective shuffle) ----------------------------
  const { split } = useSchemaDefaults();
  const holdoutDefaults = split?.holdout?.defaults ?? null;
  const kfoldDefaults = split?.kfold?.defaults ?? null;

  // --- config stores ------------------------------------------------------
  const featureCtx = useFeatureStore();
  const scaleMethod = useSettingsStore((s) => s.scaleMethod);
  const metric = useSettingsStore((s) => s.metric);

  const trainModel = useModelConfigStore((s) => s.train);

  const splitMode = useTrainingStore((s) => s.splitMode);
  const trainFrac = useTrainingStore((s) => s.trainFrac);
  const nSplits = useTrainingStore((s) => s.nSplits);
  const stratified = useTrainingStore((s) => s.stratified);
  const shuffle = useTrainingStore((s) => s.shuffle);
  const seed = useTrainingStore((s) => s.seed);
  const useShuffleBaseline = useTrainingStore((s) => s.useShuffleBaseline);
  const nShuffles = useTrainingStore((s) => s.nShuffles);

  // --- outputs ------------------------------------------------------------
  const setTrainResult = useResultsStore((s) => s.setTrainResult);
  const clearTrainResult = useResultsStore((s) => s.clearTrainResult);
  const setActiveResultKind = useResultsStore((s) => s.setActiveResultKind);

  const setArtifact = useModelArtifactStore((s) => s.setArtifact);
  const clearArtifact = useModelArtifactStore((s) => s.clearArtifact);

  // --- local runner state -------------------------------------------------
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState('');

  const pollStopRef = useRef(false);

  useEffect(() => () => {
    pollStopRef.current = true;
  }, []);

  const stopProgressPolling = useCallback(() => {
    pollStopRef.current = true;
  }, []);

  const startProgressPolling = useCallback((progressId) => {
    pollStopRef.current = false;
    async function tick() {
      if (pollStopRef.current) return;
      try {
        const rec = await fetchProgress(progressId);
        const pct = Math.round(rec?.percent || 0);
        setProgress(pct);
        setProgressLabel(rec?.label || '');

        let delay = 250;
        if (pct >= 90 && pct < 98) delay = 500;
        else if (pct >= 98 && !rec?.done) delay = 1000;

        if (!rec?.done) setTimeout(tick, delay);
        else pollStopRef.current = true;
      } catch {
        setTimeout(tick, 300);
      }
    }
    tick();
  }, []);

  const runTraining = useCallback(async () => {
    if (!dataReady) {
      setError('Load & inspect data first in the left sidebar.');
      return;
    }
    if (!trainModel) return;

    setError(null);
    clearTrainResult();
    clearArtifact();
    setIsRunning(true);

    const wantProgress = useShuffleBaseline && Number(nShuffles) > 0;
    const progressId = wantProgress ? makeProgressId() : null;
    if (wantProgress) {
      setProgress(0);
      setProgressLabel(`Shuffling 0/${nShuffles}…`);
      startProgressPolling(progressId);
    } else {
      stopProgressPolling();
      setProgress(0);
      setProgressLabel('');
    }

    try {
      const data = buildDataPayload({ xPath, yPath, npzPath, xKey, yKey });
      const scale = buildScalePayload({ method: scaleMethod });
      const features = buildFeaturesPayload(featureCtx);

      const defaultMode = getDefaultSplitMode({ split, allowedModes: ['holdout', 'kfold'] });
      const effectiveMode = splitMode ?? defaultMode;
      const defaultShuffle =
        effectiveMode === 'kfold' ? kfoldDefaults?.shuffle : holdoutDefaults?.shuffle;
      const effectiveShuffle = shuffle ?? defaultShuffle;

      const seedInt = seed === '' ? undefined : Number.parseInt(String(seed), 10);
      const evalCfg = buildEvalPayload({
        metric,
        seed: Boolean(effectiveShuffle) && Number.isFinite(seedInt) ? seedInt : undefined,
        nShuffles: wantProgress ? Number(nShuffles) : undefined,
        progressId: wantProgress ? progressId : undefined,
      });

      const splitCfg = buildSplitPayload({
        mode: effectiveMode,
        trainFrac,
        nSplits,
        stratified,
        shuffle,
        // NOTE: keep seed in eval as before.
      });

      const payload = {
        data,
        scale,
        features,
        model: trainModel,
        eval: evalCfg,
        split: splitCfg,
      };

      const resp = await runTrainRequest(payload);
      setTrainResult(resp);
      setActiveResultKind('train');
      if (resp?.artifact) setArtifact(resp.artifact);
    } catch (e) {
      setError(toErrorText(e));
      setProgress(0);
      setProgressLabel('');
    } finally {
      stopProgressPolling();
      setIsRunning(false);
    }
  }, [
    dataReady,
    trainModel,
    clearTrainResult,
    clearArtifact,
    useShuffleBaseline,
    nShuffles,
    startProgressPolling,
    stopProgressPolling,
    xPath,
    yPath,
    npzPath,
    xKey,
    yKey,
    scaleMethod,
    featureCtx,
    splitMode,
    trainFrac,
    nSplits,
    stratified,
    shuffle,
    seed,
    metric,
    split,
    holdoutDefaults?.shuffle,
    kfoldDefaults?.shuffle,
    setTrainResult,
    setActiveResultKind,
    setArtifact,
  ]);

  return {
    dataReady,
    isRunning,
    error,
    setError,
    progress,
    progressLabel,
    runTraining,
  };
}
