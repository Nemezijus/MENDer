import {
  Card, Button, Text,
  Stack, Group, Divider, Alert, Title, Box, Progress,
} from '@mantine/core';

import { useSingleModelTrainingPanelController } from '../hooks/useSingleModelTrainingPanelController.js';

import ModelSelectionCard from './ModelSelectionCard.jsx';
import ShuffleLabelsCard from './ShuffleLabelsCard.jsx';
import SplitOptionsCard from '../../../shared/ui/config/SplitOptionsCard.jsx';

import '../styles/trainingPanel.css';

/** ---------- component ---------- **/

export default function SingleModelTrainingPanel() {
  const {
    defsLoading,
    models,
    enums,
    trainModel,
    setTrainModel,

    splitMode,
    trainFrac,
    nSplits,
    stratified,
    shuffle,
    seed,
    setSeed,
    effectiveShuffleBaselineEnabled,
    effectiveNShuffles,
    defaultNShuffles,
    nShuffles,

    handleSplitModeChange,
    handleTrainFracChange,
    handleNSplitsChange,
    handleStratifiedChange,
    handleShuffleChange,
    handleShuffleBaselineCheckedChange,
    handleNShufflesChange,

    dataReady,
    isRunning,
    error,
    setError,
    progress,
    progressLabel,
    runTraining,
  } = useSingleModelTrainingPanelController();

  if (defsLoading || !models || !trainModel) {
    return null; // optionally render a skeleton
  }

  return (
    <Stack gap="lg" maw={760}>
      <Title order={3}>Run a Model</Title>

      {error && (
        <Alert color="red" title="Error" variant="light">
          <Text size="sm" className="trainingErrorText">{error}</Text>
        </Alert>
      )}

      <Card withBorder shadow="sm" radius="md" padding="lg">
        <Stack gap="md">
          <Group justify="space-between" wrap="nowrap">
            <Text fw={500}>Configuration</Text>
            <Button
              size="xs"
              onClick={() => {
                setError(null);
                runTraining();
              }}
              loading={isRunning}
              disabled={!dataReady}
            >
              {isRunning ? 'Running…' : 'Run'}
            </Button>
          </Group>

          {isRunning && effectiveShuffleBaselineEnabled && Number(effectiveNShuffles) > 0 && (
            <Stack gap={4}>
              <Group justify="space-between">
                <Text size="xs" c="dimmed">{progressLabel || 'Running…'}</Text>
                <Text size="xs" c="dimmed">{progress}%</Text>
              </Group>
              <Progress value={progress} />
            </Stack>
          )}

          {/* Centered configuration stack inside the card */}
          <Box className="trainingConfigBody">
            <Stack gap="sm">
              <SplitOptionsCard
                allowedModes={['holdout', 'kfold']}
                mode={splitMode}
                onModeChange={handleSplitModeChange}
                trainFrac={trainFrac}
                onTrainFracChange={handleTrainFracChange}
                nSplits={nSplits}
                onNSplitsChange={handleNSplitsChange}
                stratified={stratified}
                onStratifiedChange={handleStratifiedChange}
                shuffle={shuffle}
                onShuffleChange={handleShuffleChange}
                seed={seed}
                onSeedChange={setSeed}
              />

              <Divider my="xs" />

              <ModelSelectionCard
                model={trainModel}
                onChange={setTrainModel}
                schema={models?.schema}
                enums={enums}
                models={models}
                showHelp={true}
              />

              <Divider my="xs" />

              <ShuffleLabelsCard
                checked={effectiveShuffleBaselineEnabled}
                onCheckedChange={handleShuffleBaselineCheckedChange}
                nShuffles={nShuffles}
                nShufflesPlaceholder={
                  defaultNShuffles != null ? `Default: ${defaultNShuffles}` : undefined
                }
                nShufflesDescription={
                  defaultNShuffles == null
                    ? 'Schema did not provide a default number of shuffles.'
                    : undefined
                }
                onNShufflesChange={handleNShufflesChange}
              />
            </Stack>
          </Box>
        </Stack>
      </Card>
    </Stack>
  );
}
