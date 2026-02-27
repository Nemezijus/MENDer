import { Alert, Text } from '@mantine/core';

import { useDataStore } from '../../features/dataFiles/state/useDataStore.js';

export default function DataGuard({ children }) {
  const dataReady = useDataStore(
    (s) => !!s.inspectReport && s.inspectReport.n_samples > 0,
  );

  if (!dataReady) {
    return (
      <Alert color="yellow" variant="light">
        <Text fw={500}>No inspected training data yet.</Text>
        <Text size="sm">
          Please upload and inspect your training data in the{' '}
          <strong>Data &amp; files</strong> section before using this panel.
        </Text>
      </Alert>
    );
  }

  return children;
}
