import { Card, Stack, Text } from '@mantine/core';
import { useResultsStore } from '../state/useResultsStore.js';
import ModelTrainingResultsPanel from './ModelTrainingResultsPanel.jsx';

export default function ResultsPanel() {
  const activeResultKind = useResultsStore((s) => s.activeResultKind);
  const hasTrainResult = useResultsStore((s) => !!s.trainResult);

  // For now we only support "train" results; default to that if set.
  if (activeResultKind === 'train' || (!activeResultKind && hasTrainResult)) {
    return <ModelTrainingResultsPanel />;
  }

  // Generic fallback: no results selected
  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="xs">
        <Text fw={500}>Results</Text>
        <Text size="sm" c="dimmed">
          No results to display yet. Run a model (or another analysis) and the
          latest results will appear here.
        </Text>
      </Stack>
    </Card>
  );
}
