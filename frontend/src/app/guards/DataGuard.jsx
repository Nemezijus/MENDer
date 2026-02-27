import { Alert, Text } from '@mantine/core';

import '../styles/guards.css';

import { useDataStore } from '../../features/dataFiles/state/useDataStore.js';

export default function DataGuard({ children }) {
  const dataReady = useDataStore(
    (s) => !!s.inspectReport && s.inspectReport.n_samples > 0,
  );

  if (!dataReady) {
    return (
      <Alert color="yellow" variant="light">
        <Text className="dataGuardTitle">No inspected training data yet.</Text>
        <Text className="dataGuardBody">
          Please upload and inspect your training data in the{' '}
          <strong>Data &amp; files</strong> section before using this panel.
        </Text>
      </Alert>
    );
  }

  return children;
}
