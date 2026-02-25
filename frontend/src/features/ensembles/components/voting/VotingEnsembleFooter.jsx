import { Group, Button, Alert, Text, Stack } from '@mantine/core';

export default function VotingEnsembleFooter({ onRun, loading, disabled }) {
  return (
    <Stack gap="md">
      <Group justify="flex-end">
        <Button onClick={onRun} loading={loading} disabled={disabled}>
          Train voting ensemble
        </Button>
      </Group>

      <Alert color="blue" variant="light">
        <Text size="sm">
          This uses your current <strong>global</strong> Scaling / Metric / Features settings from the Settings section.
        </Text>
      </Alert>
    </Stack>
  );
}
