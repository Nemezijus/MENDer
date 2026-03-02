import { useEffect, useMemo, useState } from 'react';
import { Stack, Card, Text, Group, Button, Alert, Title } from '@mantine/core';
import { useShallow } from 'zustand/react/shallow';

import { useSchemaDefaults } from '../../../shared/schema/SchemaDefaultsContext.jsx';
import { useDataStore } from '../../dataFiles/state/useDataStore.js';
import { useFeatureStore } from '../../../shared/state/useFeatureStore.js';
import { useUnsupervisedStore } from '../state/useUnsupervisedStore.js';

import ModelSelectionCard from '../../training/components/ModelSelectionCard.jsx';
import { useUnsupervisedTrainer } from '../hooks/useUnsupervisedTrainer.js';

import '../styles/unsupervisedPanel.css';

export default function UnsupervisedTrainingPanel() {
  const schema = useSchemaDefaults();

  // Data selections come from the global Upload tab.
  const { xPath, npzPath } = useDataStore(
    useShallow((s) => ({
      xPath: s.xPath,
      npzPath: s.npzPath,
    })),
  );

  // Features method (global Settings tab).
  const { method: featureMethod, setMethod } = useFeatureStore(
    useShallow((s) => ({
      method: s.method,
      setMethod: s.setMethod,
    })),
  );

  const { model, algo, setModel } = useUnsupervisedStore(
    useShallow((s) => ({
      model: s.model,
      algo: s.algo,
      setModel: s.setModel,
    })),
  );

  const { loading: isRunning, error, clearError, runTraining } = useUnsupervisedTrainer();

  const [clearedFeature, setClearedFeature] = useState(null);

  // Available unsupervised algorithms from backend meta.
  const unsupAlgos = useMemo(() => {
    return schema.getCompatibleAlgos?.('unsupervised') || [];
  }, [schema]);

  // Initialize model once (and keep selections across navigation).
  useEffect(() => {
    if (model?.algo) return;
    if (!unsupAlgos || unsupAlgos.length === 0) return;

    const preferredAlgo = algo || unsupAlgos[0];
    setModel({ algo: preferredAlgo });
  }, [algo, model?.algo, setModel, unsupAlgos]);

  const hasX = !!(xPath || npzPath);
  const canRun = hasX && !!model?.algo && !schema.loading && !isRunning;

  // IMPORTANT: the Engine unsupervised pipeline fits on X only (y is ignored), so
  // supervised feature methods (LDA/SFS) are not supported here.
  // Cleaner behavior: if a user has an override set to an incompatible method, clear it.
  useEffect(() => {
    const isSupervisedOnly = featureMethod === 'lda' || featureMethod === 'sfs';
    if (!isSupervisedOnly) return;
    setMethod(undefined);
    setClearedFeature(featureMethod);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [featureMethod]);

  const handleRun = async () => {
    clearError();
    await runTraining();
  };

  return (
    <Stack gap="md">
      <Title order={3}>Train an unsupervised Model</Title>

      <Card withBorder shadow="sm" radius="md" padding="md">
        <Stack gap="sm">
          <Group justify="space-between" align="flex-start" wrap="wrap">
            <Text fw={600}>Configuration</Text>
            <Button size="xs" onClick={handleRun} disabled={!canRun} loading={isRunning}>
              Run
            </Button>
          </Group>

          {!hasX ? (
            <Alert color="red" title="Missing Feature matrix (X)">
              Select a Feature matrix (X) in <b>Upload data &amp; models</b>.
            </Alert>
          ) : null}

          {clearedFeature ? (
            <Alert color="yellow" title="Incompatible Features method cleared">
              The Features method <b>{clearedFeature}</b> is supervised-only and is not supported for unsupervised
              training (the Engine fits on <b>X only</b>). The override was cleared. Use <b>PCA</b> or <b>None</b> in the
              <b> Settings</b> tab.
            </Alert>
          ) : null}

          {error ? (
            <Alert color="red" title="Error">
              <Text size="sm" className="unsupPreWrapText">
                {error}
              </Text>
            </Alert>
          ) : null}

          <ModelSelectionCard
            model={model}
            onChange={setModel}
            schema={schema.models?.schema}
            enums={schema.enums}
            models={schema.models}
            taskOverride="unsupervised"
            showHelp
          />
        </Stack>
      </Card>

      <Text size="sm" c="dimmed">
        This uses your current global Scaling / Features settings from the Settings section.
      </Text>
    </Stack>
  );
}
