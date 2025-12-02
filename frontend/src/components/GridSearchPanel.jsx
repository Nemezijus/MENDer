import { useEffect, useState } from 'react';
import {
  Card,
  Button,
  Text,
  Stack,
  Group,
  Divider,
  Alert,
  Title,
  Box,
  NumberInput,
} from '@mantine/core';

import { useDataStore } from '../state/useDataStore.js';
import { useFeatureStore } from '../state/useFeatureStore.js';
import { useSettingsStore } from '../state/useSettingsStore.js';
import { useSchemaDefaults } from '../state/SchemaDefaultsContext';
import { useTuningStore } from '../state/useTuningStore.js';
import { useModelConfigStore } from '../state/useModelConfigStore.js';

import SplitOptionsCard from './SplitOptionsCard.jsx';
import ModelSelectionCard from './ModelSelectionCard.jsx';
import { requestGridSearch } from '../api/tuning';
import GridSearchResultsPanel from './visualizations/GridSearchResultsPanel.jsx';
import HyperparameterSelector from './helpers/HyperparameterSelector.jsx';

const EMPTY_PARAM = { paramName: '', values: [] };

export default function GridSearchPanel() {
  const xPath = useDataStore((s) => s.xPath);
  const yPath = useDataStore((s) => s.yPath);
  const npzPath = useDataStore((s) => s.npzPath);
  const xKey = useDataStore((s) => s.xKey);
  const yKey = useDataStore((s) => s.yKey);
  const inspectReport = useDataStore((s) => s.inspectReport);
  const dataReady = !!inspectReport && inspectReport?.n_samples > 0;

  const {
    method,
    pca_n,
    pca_var,
    pca_whiten,
    lda_n,
    lda_solver,
    lda_shrinkage,
    lda_tol,
    sfs_k,
    sfs_direction,
    sfs_cv,
    sfs_n_jobs,
  } = useFeatureStore();

  const { loading: defsLoading, models, enums, getModelDefaults } = useSchemaDefaults();

  const scaleMethod = useSettingsStore((s) => s.scaleMethod);
  const metric = useSettingsStore((s) => s.metric);

  const gsState = useTuningStore((s) => s.gridSearch);
  const setGsState = useTuningStore((s) => s.setGridSearch);

  const {
    nSplits,
    stratified,
    shuffle,
    seed,
    nJobs,
    result: gridResult,
  } = gsState;

  // two hyperparameters to vary
  const hyperParam1 = gsState.hyperParam1 ?? EMPTY_PARAM;
  const hyperParam2 = gsState.hyperParam2 ?? EMPTY_PARAM;

  const handleHyperParam1Change = (next) =>
    setGsState({ hyperParam1: next });

  const handleHyperParam2Change = (next) =>
    setGsState({ hyperParam2: next });

  // Per-panel model config (grid search slice)
  const gsModel = useModelConfigStore((s) => s.gridSearch);
  const setGsModel = useModelConfigStore((s) => s.setGridSearchModel);

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  // Initialize GS model once
  useEffect(() => {
    if (!defsLoading && !gsModel) {
      const init = getModelDefaults('logreg') || { algo: 'logreg' };
      setGsModel(init);
    }
  }, [defsLoading, getModelDefaults, gsModel, setGsModel]);

  async function handleCompute() {
    if (!dataReady) {
      setErr('Load & inspect data first in the Data & files section.');
      return;
    }
    if (!gsModel) return;

    const p1 = (hyperParam1.paramName || '').trim();
    const p2 = (hyperParam2.paramName || '').trim();
    const v1 = Array.isArray(hyperParam1.values) ? hyperParam1.values : [];
    const v2 = Array.isArray(hyperParam2.values) ? hyperParam2.values : [];

    if (!p1 || !p2) {
      setErr('Please select exactly two hyperparameters to vary.');
      return;
    }
    if (p1 === p2) {
      setErr('The two hyperparameters must be different.');
      return;
    }
    if (v1.length < 2 || v2.length < 2) {
      setErr('Each hyperparameter must have at least two values.');
      return;
    }

    setErr(null);
    setGsState({ result: null });
    setLoading(true);

    try {
      // param_grid shaped for the backend; adjust if your service expects a different structure
      const paramGrid = {
        [p1]: v1,
        [p2]: v2,
      };

      const payload = {
        data: {
          x_path: npzPath ? null : xPath,
          y_path: npzPath ? null : yPath,
          npz_path: npzPath,
          x_key: xKey,
          y_key: yKey,
        },
        split: {
          mode: 'kfold',
          n_splits: nSplits,
          stratified,
          shuffle,
        },
        scale: { method: scaleMethod },
        features: {
          method,
          pca_n,
          pca_var,
          pca_whiten,
          lda_n,
          lda_solver,
          lda_shrinkage,
          lda_tol,
          sfs_k: sfs_k === '' || sfs_k == null ? 'auto' : sfs_k,
          sfs_direction,
          sfs_cv,
          sfs_n_jobs,
        },
        model: gsModel,
        eval: {
          metric,
          seed: shuffle ? (seed === '' ? null : parseInt(seed, 10)) : null,
        },
        param_grid: paramGrid,
        n_jobs: Number(nJobs),
      };

      const data = await requestGridSearch(payload);
      // expect backend to include metric_used & 2D grid info; we store metric here as well
      setGsState({ result: { ...data, metric_used: metric } });
    } catch (e) {
      const raw = e?.response?.data?.detail ?? e.message ?? String(e);
      const msg = typeof raw === 'string' ? raw : JSON.stringify(raw, null, 2);
      setErr(msg);
    } finally {
      setLoading(false);
    }
  }

  if (defsLoading || !models || !gsModel) {
    return null;
  }

  return (
    <Stack gap="md">
      <Title order={3}>Grid search</Title>

      {err && (
        <Alert color="red" variant="light">
          <Text fw={500}>Error</Text>
          <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
            {err}
          </Text>
        </Alert>
      )}

      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Group justify="space-between" wrap="nowrap">
            <Text fw={500}>Configuration</Text>
            <Button
              size="xs"
              onClick={handleCompute}
              loading={loading}
              disabled={!dataReady}
            >
              {loading ? 'Searching…' : 'Run grid search'}
            </Button>
          </Group>

          <Box
            style={{
              maxWidth: 560,
              margin: '0 auto',
              width: '100%',
            }}
          >
            <Stack gap="sm">
              <SplitOptionsCard
                allowedModes={['kfold']}
                nSplits={nSplits}
                onNSplitsChange={(value) => setGsState({ nSplits: value })}
                stratified={stratified}
                onStratifiedChange={(value) => setGsState({ stratified: value })}
                shuffle={shuffle}
                onShuffleChange={(value) => setGsState({ shuffle: value })}
                seed={seed}
                onSeedChange={(value) => setGsState({ seed: value })}
              />

              <Divider my="xs" />

              <ModelSelectionCard
                model={gsModel}
                onChange={(next) => {
                  if (next?.algo && gsModel && next.algo !== gsModel.algo) {
                    const d = getModelDefaults(next.algo) || { algo: next.algo };
                    setGsModel({ ...d, ...next });
                    // reset hyperparameters on algo change
                    setGsState({
                      hyperParam1: EMPTY_PARAM,
                      hyperParam2: EMPTY_PARAM,
                    });
                  } else {
                    setGsModel(next);
                  }
                }}
                schema={models?.schema}
                enums={enums}
                models={models}
              />

              <Divider my="xs" />

              <Stack gap="sm">
                <Text size="sm" fw={500}>
                  Select the 1st parameter to vary:
                </Text>
                <HyperparameterSelector
                  schema={models?.schema}
                  model={gsModel}
                  value={hyperParam1}
                  onChange={handleHyperParam1Change}
                  label="1st hyperparameter"
                />
              </Stack>

              <Stack gap="sm">
                <Text size="sm" fw={500}>
                  Select the 2nd parameter to vary:
                </Text>
                <HyperparameterSelector
                  schema={models?.schema}
                  model={gsModel}
                  value={hyperParam2}
                  onChange={handleHyperParam2Change}
                  label="2nd hyperparameter"
                />
              </Stack>

              <Box style={{ maxWidth: 180 }}>
                <NumberInput
                  label="n_jobs"
                  min={1}
                  step={1}
                  value={nJobs}
                  onChange={(value) => setGsState({ nJobs: value })}
                />
              </Box>
            </Stack>
          </Box>

          {loading && (
            <Text size="xs" c="dimmed">
              Results will appear once the grid search finishes.
            </Text>
          )}
        </Stack>
      </Card>

      {gridResult && <GridSearchResultsPanel result={gridResult} />}
    </Stack>
  );
}
