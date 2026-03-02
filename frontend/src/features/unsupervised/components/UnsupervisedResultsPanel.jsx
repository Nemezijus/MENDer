import { Stack } from '@mantine/core';

import UnsupervisedDiagnosticsCard from './diagnostics/UnsupervisedDiagnosticsCard.jsx';
import UnsupervisedDecoderOutputsCard from './decoder/UnsupervisedDecoderOutputsCard.jsx';

import '../styles/unsupervisedPanel.css';
import '../styles/diagnostics.css';
import '../styles/previewTable.css';

export default function UnsupervisedResultsPanel({ trainResult }) {
  if (!trainResult) return null;

  return (
    <Stack gap="md">
      <UnsupervisedDiagnosticsCard trainResult={trainResult} />
      <UnsupervisedDecoderOutputsCard trainResult={trainResult} />
    </Stack>
  );
}
